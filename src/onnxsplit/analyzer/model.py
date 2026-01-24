"""ONNX模型分析器"""

from pathlib import Path
from typing import Optional

import onnx
from onnx import ModelProto, ValueInfoProto

from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.analyzer.tensor import TensorMetadata


class ModelAnalyzer:
    """ONNX模型分析器

    提供模型解析、形状推断、算子信息提取等功能。
    """

    def __init__(self, model: ModelProto):
        """初始化分析器

        Args:
            model: ONNX模型
        """
        self.model = model
        self.graph = model.graph
        self._shape_map: dict[str, tuple[int, ...]] = {}
        self._dtype_map: dict[str, int] = {}
        self._build_tensor_info()

    @classmethod
    def from_path(cls, path: Path | str) -> "ModelAnalyzer":
        """从文件路径加载模型

        Args:
            path: ONNX模型文件路径

        Returns:
            ModelAnalyzer实例
        """
        model = onnx.load(str(path))
        return cls(model)

    @classmethod
    def from_model_proto(cls, model: ModelProto) -> "ModelAnalyzer":
        """从ModelProto创建分析器

        Args:
            model: ONNX模型

        Returns:
            ModelAnalyzer实例
        """
        return cls(model)

    def _build_tensor_info(self) -> None:
        """构建张量形状和类型信息映射"""
        # 从输入获取
        for value_info in self.graph.input:
            self._add_tensor_info(value_info)

        # 从输出获取
        for value_info in self.graph.output:
            self._add_tensor_info(value_info)

        # 从value_info获取（如果模型有形状信息）
        for value_info in self.graph.value_info:
            self._add_tensor_info(value_info)

    def _add_tensor_info(self, value_info: ValueInfoProto) -> None:
        """添加张量信息到映射表"""
        name = value_info.name
        if value_info.type.tensor_type:
            shape = tuple(
                d.dim_value if d.dim_value > 0 else -1
                for d in value_info.type.tensor_type.shape.dim
            )
            dtype = value_info.type.tensor_type.elem_type
            self._shape_map[name] = shape
            self._dtype_map[name] = dtype

    def _get_tensor_shape(self, name: str) -> tuple[int, ...]:
        """获取张量形状"""
        return self._shape_map.get(name, ())

    def _get_tensor_dtype(self, name: str) -> int:
        """获取张量数据类型"""
        return self._dtype_map.get(name, onnx.TensorProto.UNDEFINED)

    def get_inputs(self) -> list[TensorMetadata]:
        """获取模型输入信息

        Returns:
            输入张量元数据列表
        """
        inputs = []
        for value_info in self.graph.input:
            shape = self._get_tensor_shape(value_info.name)
            dtype = self._get_tensor_dtype(value_info.name)
            inputs.append(TensorMetadata(value_info.name, shape, dtype))
        return inputs

    def get_outputs(self) -> list[TensorMetadata]:
        """获取模型输出信息

        Returns:
            输出张量元数据列表
        """
        outputs = []
        for value_info in self.graph.output:
            shape = self._get_tensor_shape(value_info.name)
            dtype = self._get_tensor_dtype(value_info.name)
            outputs.append(TensorMetadata(value_info.name, shape, dtype))
        return outputs

    def get_operators(self) -> list[OperatorInfo]:
        """获取所有算子信息

        跳过Constant算子（通常是权重）。

        Returns:
            算子信息列表
        """
        operators = []

        for node in self.graph.node:
            # 跳过常量
            if node.op_type == "Constant":
                continue

            op_info = OperatorInfo.from_node_proto(node)

            # 添加输入张量信息
            for input_name in node.input:
                if not input_name:  # 空输入（可选输入）
                    continue
                shape = self._get_tensor_shape(input_name)
                dtype = self._get_tensor_dtype(input_name)
                if shape:  # 只有已知形状才添加
                    op_info.input_tensors.append(TensorMetadata(input_name, shape, dtype))

            # 添加输出张量信息
            for output_name in node.output:
                shape = self._get_tensor_shape(output_name)
                dtype = self._get_tensor_dtype(output_name)
                if shape:
                    op_info.output_tensors.append(TensorMetadata(output_name, shape, dtype))

            operators.append(op_info)

        return operators

    def get_operator(self, name: str) -> Optional[OperatorInfo]:
        """按名称获取算子

        Args:
            name: 算子名称

        Returns:
            算子信息，不存在时返回None
        """
        for op in self.get_operators():
            if op.name == name:
                return op
        return None

    def get_tensor_producer(self, tensor_name: str) -> Optional[str]:
        """获取产生指定张量的算子名称

        Args:
            tensor_name: 张量名称

        Returns:
            产生该张量的算子名称，如果是图输入则返回None
        """
        for node in self.graph.node:
            if tensor_name in node.output:
                if node.name:
                    return node.name
                if node.output:  # 防止空列表导致IndexError
                    return f"{node.op_type}_{node.output[0]}"
                return f"{node.op_type}_unknown"
        return None

    def get_tensor_consumers(self, tensor_name: str) -> list[str]:
        """获取使用指定张量的算子名称列表

        Args:
            tensor_name: 张量名称

        Returns:
            使用该张量的算子名称列表
        """
        consumers = []
        for node in self.graph.node:
            if tensor_name in node.input:
                if node.name:
                    name = node.name
                elif node.output:  # 防止空列表导致IndexError
                    name = f"{node.op_type}_{node.output[0]}"
                else:
                    name = f"{node.op_type}_unknown"
                consumers.append(name)
        return consumers

    @property
    def ir_version(self) -> int:
        """获取模型IR版本"""
        return self.model.ir_version

    @property
    def opset_version(self) -> int:
        """获取opset版本"""
        if self.model.opset_import:
            return self.model.opset_import[0].version
        return 0

    @property
    def producer_name(self) -> str:
        """获取模型生产者名称"""
        return self.model.producer_name or ""

    @property
    def producer_version(self) -> str:
        """获取模型生产者版本"""
        return self.model.producer_version or ""

    @property
    def graph_name(self) -> str:
        """获取图名称"""
        return self.graph.name or ""

    def __repr__(self) -> str:
        return (
            f"ModelAnalyzer(graph={self.graph_name!r}, "
            f"opset={self.opset_version}, "
            f"operators={len(self.get_operators())})"
        )
