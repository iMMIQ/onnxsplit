"""内存估算器"""

from dataclasses import dataclass
from typing import Optional

from onnx import TensorProto

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo

_DTYPE_SIZE_MAP = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.INT8: 1,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.UINT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.COMPLEX64: 8,
    TensorProto.COMPLEX128: 16,
}


def dtype_bytes(dtype: int) -> int:
    """获取数据类型的字节大小"""
    return _DTYPE_SIZE_MAP.get(dtype, 4)


def estimate_tensor_memory(shape: tuple[int, ...], dtype: int) -> int:
    """估算张量内存占用

    Args:
        shape: 张量形状
        dtype: 数据类型

    Returns:
        内存字节数，包含动态维度时返回0
    """
    if any(s < 0 for s in shape if s != 0):
        return 0

    if not shape:
        return dtype_bytes(dtype)

    numel = 1
    for dim in shape:
        numel *= dim

    return numel * dtype_bytes(dtype)


@dataclass
class TensorMemoryInfo:
    """张量内存信息"""

    tensor_name: str
    shape: tuple[int, ...]
    dtype: int
    memory_bytes: int

    @property
    def size_mb(self) -> float:
        """内存大小（MB）"""
        return self.memory_bytes / (1024 * 1024)

    @property
    def dtype_name(self) -> str:
        """数据类型名称"""
        return TensorProto.DataType.Name(self.dtype)


@dataclass
class OperatorMemoryInfo:
    """算子内存信息"""

    operator_name: str
    op_type: str
    input_memory_mb: float
    output_memory_mb: float
    weights_memory_mb: float
    total_memory_mb: float
    peak_memory_mb: float  # 执行期间的峰值内存


class MemoryEstimator:
    """内存估算器

    估算模型和算子的内存占用。
    """

    def __init__(self, analyzer: ModelAnalyzer):
        """初始化估算器

        Args:
            analyzer: 模型分析器
        """
        self.analyzer = analyzer
        self._tensor_memory: dict[str, TensorMemoryInfo] = {}
        self._operator_memory: dict[str, OperatorMemoryInfo] = {}
        self._build_memory_info()

    def _build_memory_info(self) -> None:
        """构建内存信息"""
        # 收集所有张量的内存信息
        for value_info in self.analyzer.model.graph.input:
            self._add_tensor_info(value_info.name, value_info)

        for value_info in self.analyzer.model.graph.output:
            self._add_tensor_info(value_info.name, value_info)

        for value_info in self.analyzer.model.graph.value_info:
            self._add_tensor_info(value_info.name, value_info)

        # 计算算子内存
        for op_info in self.analyzer.get_operators():
            self._calculate_operator_memory(op_info)

    def _add_tensor_info(self, name: str, value_info) -> None:
        """添加张量内存信息"""
        if value_info.type.tensor_type:
            shape = tuple(
                d.dim_value if d.dim_value > 0 else -1
                for d in value_info.type.tensor_type.shape.dim
            )
            dtype = value_info.type.tensor_type.elem_type
            memory_bytes = estimate_tensor_memory(shape, dtype)

            self._tensor_memory[name] = TensorMemoryInfo(
                tensor_name=name,
                shape=shape,
                dtype=dtype,
                memory_bytes=memory_bytes,
            )

    def _calculate_operator_memory(self, op_info: OperatorInfo) -> None:
        """计算算子内存"""
        input_memory = 0
        output_memory = 0
        weights_memory = 0

        # 输入内存
        for tensor in op_info.input_tensors:
            if tensor.name not in self._tensor_memory:
                mem = estimate_tensor_memory(tensor.shape, tensor.dtype)
                self._tensor_memory[tensor.name] = TensorMemoryInfo(
                    tensor_name=tensor.name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    memory_bytes=mem,
                )
            input_memory += self._tensor_memory[tensor.name].memory_bytes

        # 输出内存
        for tensor in op_info.output_tensors:
            if tensor.name not in self._tensor_memory:
                mem = estimate_tensor_memory(tensor.shape, tensor.dtype)
                self._tensor_memory[tensor.name] = TensorMemoryInfo(
                    tensor_name=tensor.name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    memory_bytes=mem,
                )
            output_memory += self._tensor_memory[tensor.name].memory_bytes

        # 检查是否有权重输入
        for input_name in op_info.input_names:
            if self._is_weight(input_name):
                if input_name in self._tensor_memory:
                    weights_memory += self._tensor_memory[input_name].memory_bytes

        total_memory = input_memory + output_memory + weights_memory

        self._operator_memory[op_info.name] = OperatorMemoryInfo(
            operator_name=op_info.name,
            op_type=op_info.op_type,
            input_memory_mb=input_memory / (1024 * 1024),
            output_memory_mb=output_memory / (1024 * 1024),
            weights_memory_mb=weights_memory / (1024 * 1024),
            total_memory_mb=total_memory / (1024 * 1024),
            peak_memory_mb=total_memory / (1024 * 1024),  # 简化估算
        )

    def _is_weight(self, tensor_name: str) -> bool:
        """检查是否是权重张量"""
        return any(init.name == tensor_name for init in self.analyzer.model.graph.initializer)

    def get_tensor_memory(self, tensor_name: str) -> Optional[TensorMemoryInfo]:
        """获取张量内存信息"""
        return self._tensor_memory.get(tensor_name)

    def get_operator_memory(self, op_info: OperatorInfo) -> Optional[OperatorMemoryInfo]:
        """获取算子内存信息"""
        return self._operator_memory.get(op_info.name)

    def get_total_model_memory(self) -> int:
        """获取模型总内存（字节）"""
        return sum(info.memory_bytes for info in self._tensor_memory.values())

    def get_peak_memory(self) -> float:
        """获取峰值内存（MB）"""
        if not self._operator_memory:
            return 0.0
        return max(info.peak_memory_mb for info in self._operator_memory.values())

    def get_memory_breakdown(self) -> list[OperatorMemoryInfo]:
        """获取内存分解"""
        return list(self._operator_memory.values())

    def get_weights_memory(self) -> int:
        """获取权重总内存（字节）"""
        total = 0
        for init in self.analyzer.model.graph.initializer:
            # 计算初始器大小
            dims = list(init.dims)
            if dims:
                numel = 1
                for dim in dims:
                    numel *= dim
                total += numel * dtype_bytes(init.data_type)
        return total
