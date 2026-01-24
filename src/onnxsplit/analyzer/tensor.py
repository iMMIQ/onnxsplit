"""张量元数据结构"""
from dataclasses import dataclass
from onnx import TensorProto


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


def dtype_to_bytes(dtype: int) -> int:
    """获取数据类型的字节大小

    Args:
        dtype: ONNX TensorProto 数据类型常量

    Returns:
        该类型每个元素的字节大小，未知类型默认返回4
    """
    return _DTYPE_SIZE_MAP.get(dtype, 4)


@dataclass(frozen=True)
class TensorMetadata:
    """张量元数据

    Attributes:
        name: 张量名称
        shape: 张量形状（每个维度的大小）
        dtype: ONNX数据类型
    """
    name: str
    shape: tuple[int, ...]
    dtype: int

    @property
    def rank(self) -> int:
        """张量的秩（维度数量）"""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """张量中元素的总数量"""
        if not self.shape:
            return 1
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def memory_bytes(self) -> int:
        """张量占用的内存字节数"""
        return self.numel * dtype_to_bytes(self.dtype)

    @property
    def size_mb(self) -> float:
        """张量占用的内存大小（MB）"""
        return self.memory_bytes / (1024 * 1024)

    def __repr__(self) -> str:
        dtype_name = TensorProto.DataType.Name(self.dtype)
        return f"TensorMetadata(name={self.name!r}, shape={self.shape}, dtype={dtype_name})"
