"""算子信息结构"""

from dataclasses import dataclass, field
from typing import Any

from onnx import NodeProto

from onnxsplit.analyzer.tensor import TensorMetadata


@dataclass
class OperatorInfo:
    """算子信息

    Attributes:
        name: 算子名称
        op_type: 算子类型（如Conv, MatMul等）
        attributes: 算子属性字典
        input_tensors: 输入张量元数据列表
        output_tensors: 输出张量元数据列表
        input_names: 输入张量名称列表（可选）
        output_names: 输出张量名称列表（可选）
    """

    name: str
    op_type: str
    attributes: dict[str, Any]
    input_tensors: list[TensorMetadata]
    output_tensors: list[TensorMetadata]
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)

    @property
    def input_memory_mb(self) -> float:
        """输入张量总内存（MB），动态形状返回0"""
        total = 0
        for tensor in self.input_tensors:
            # 跳过包含动态维度的张量
            if any(d < 0 for d in tensor.shape if d != 0):
                continue
            total += tensor.memory_bytes
        return total / (1024 * 1024)

    @property
    def output_memory_mb(self) -> float:
        """输出张量总内存（MB），动态形状返回0"""
        total = 0
        for tensor in self.output_tensors:
            if any(d < 0 for d in tensor.shape if d != 0):
                continue
            total += tensor.memory_bytes
        return total / (1024 * 1024)

    @property
    def total_memory_mb(self) -> float:
        """算子总内存占用（MB），输入+输出"""
        return self.input_memory_mb + self.output_memory_mb

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """获取算子属性

        Args:
            key: 属性名
            default: 默认值

        Returns:
            属性值，不存在时返回默认值
        """
        return self.attributes.get(key, default)

    def get_input_shape(self, index: int) -> tuple[int, ...] | None:
        """获取指定输入的形状

        Args:
            index: 输入索引

        Returns:
            形状元组，索引越界时返回None
        """
        if 0 <= index < len(self.input_tensors):
            return self.input_tensors[index].shape
        return None

    def get_output_shape(self, index: int) -> tuple[int, ...] | None:
        """获取指定输出的形状

        Args:
            index: 输出索引

        Returns:
            形状元组，索引越界时返回None
        """
        if 0 <= index < len(self.output_tensors):
            return self.output_tensors[index].shape
        return None

    @classmethod
    def from_node_proto(cls, node: NodeProto) -> "OperatorInfo":
        """从ONNX NodeProto创建算子信息（不含形状信息）

        Args:
            node: ONNX算子节点

        Returns:
            OperatorInfo实例
        """
        attributes = {}
        for attr in node.attribute:
            if attr.name == "axes":
                attributes[attr.name] = list(attr.ints)
            elif attr.name == "kernel_shape":
                attributes[attr.name] = list(attr.ints)
            elif attr.name == "pads":
                attributes[attr.name] = list(attr.ints)
            elif attr.name == "strides":
                attributes[attr.name] = list(attr.ints)
            elif attr.type == 0:  # UNDEFINED attribute
                attributes[attr.name] = None
            elif attr.type == 1:  # FLOAT
                attributes[attr.name] = attr.f
            elif attr.type == 2:  # INT
                attributes[attr.name] = attr.i
            elif attr.type == 3:  # STRING
                attributes[attr.name] = attr.s.decode("utf-8")
            elif attr.type == 4:  # TENSOR
                attributes[attr.name] = attr.t
            elif attr.type == 5:  # GRAPH
                attributes[attr.name] = attr.g
            elif attr.type == 6:  # FLOATS
                attributes[attr.name] = list(attr.floats)
            elif attr.type == 7:  # INTS
                attributes[attr.name] = list(attr.ints)
            elif attr.type == 8:  # STRINGS
                attributes[attr.name] = [s.decode("utf-8") for s in attr.strings]

        return cls(
            name=node.name or f"{node.op_type}_{node.output[0]}",
            op_type=node.op_type,
            attributes=attributes,
            input_tensors=[],
            output_tensors=[],
            input_names=list(node.input),
            output_names=list(node.output),
        )

    def __repr__(self) -> str:
        return f"OperatorInfo(name={self.name!r}, op_type={self.op_type!r})"
