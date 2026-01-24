"""测试张量元数据结构"""
from onnx import TensorProto

from onnxsplit.analyzer.tensor import TensorMetadata, dtype_to_bytes


def test_tensor_metadata_creation():
    """测试创建张量元数据"""
    metadata = TensorMetadata(
        name="input",
        shape=(1, 3, 224, 224),
        dtype=TensorProto.FLOAT,
    )
    assert metadata.name == "input"
    assert metadata.shape == (1, 3, 224, 224)
    assert metadata.dtype == TensorProto.FLOAT


def test_tensor_metadata_memory_float():
    """测试FLOAT类型张量内存计算"""
    metadata = TensorMetadata(
        name="float_tensor",
        shape=(2, 3, 4),
        dtype=TensorProto.FLOAT,
    )
    # 2 * 3 * 4 * 4 bytes = 96 bytes
    assert metadata.memory_bytes == 96


def test_tensor_metadata_memory_float16():
    """测试FLOAT16类型张量内存计算"""
    metadata = TensorMetadata(
        name="float16_tensor",
        shape=(10, 20),
        dtype=TensorProto.FLOAT16,
    )
    # 10 * 20 * 2 bytes = 400 bytes
    assert metadata.memory_bytes == 400


def test_tensor_metadata_memory_int64():
    """测试INT64类型张量内存计算"""
    metadata = TensorMetadata(
        name="int64_tensor",
        shape=(5, 5),
        dtype=TensorProto.INT64,
    )
    # 5 * 5 * 8 bytes = 200 bytes
    assert metadata.memory_bytes == 200


def test_tensor_metadata_memory_bool():
    """测试BOOL类型张量内存计算"""
    metadata = TensorMetadata(
        name="bool_tensor",
        shape=(100,),
        dtype=TensorProto.BOOL,
    )
    # 100 * 1 byte = 100 bytes
    assert metadata.memory_bytes == 100


def test_tensor_metadata_rank():
    """测试张量秩计算"""
    metadata = TensorMetadata(
        name="tensor",
        shape=(1, 3, 224, 224),
        dtype=TensorProto.FLOAT,
    )
    assert metadata.rank == 4


def test_tensor_metadata_numel():
    """测试张量元素数量计算"""
    metadata = TensorMetadata(
        name="tensor",
        shape=(2, 3, 4),
        dtype=TensorProto.FLOAT,
    )
    assert metadata.numel == 24


def test_tensor_metadata_size_mb():
    """测试MB单位内存计算"""
    metadata = TensorMetadata(
        name="tensor",
        shape=(1024, 1024),  # 1M elements
        dtype=TensorProto.FLOAT,
    )
    # 1M * 4 bytes = 4MB
    assert metadata.size_mb == 4.0


def test_tensor_metadata_empty_shape():
    """测试标量张量（空shape）"""
    metadata = TensorMetadata(
        name="scalar",
        shape=(),
        dtype=TensorProto.FLOAT,
    )
    assert metadata.rank == 0
    assert metadata.numel == 1
    assert metadata.memory_bytes == 4


def test_dtype_to_bytes_float():
    """测试FLOAT类型字节大小"""
    assert dtype_to_bytes(TensorProto.FLOAT) == 4


def test_dtype_to_bytes_float16():
    """测试FLOAT16类型字节大小"""
    assert dtype_to_bytes(TensorProto.FLOAT16) == 2


def test_dtype_to_bytes_double():
    """测试DOUBLE类型字节大小"""
    assert dtype_to_bytes(TensorProto.DOUBLE) == 8


def test_dtype_to_bytes_int32():
    """测试INT32类型字节大小"""
    assert dtype_to_bytes(TensorProto.INT32) == 4


def test_dtype_to_bytes_int64():
    """测试INT64类型字节大小"""
    assert dtype_to_bytes(TensorProto.INT64) == 8


def test_dtype_to_bytes_int8():
    """测试INT8类型字节大小"""
    assert dtype_to_bytes(TensorProto.INT8) == 1


def test_dtype_to_bytes_uint8():
    """测试UINT8类型字节大小"""
    assert dtype_to_bytes(TensorProto.UINT8) == 1


def test_dtype_to_bytes_bool():
    """测试BOOL类型字节大小"""
    assert dtype_to_bytes(TensorProto.BOOL) == 1


def test_dtype_to_bytes_unknown():
    """测试未知类型默认4字节"""
    assert dtype_to_bytes(999) == 4


def test_tensor_metadata_repr():
    """测试张量元数据字符串表示"""
    metadata = TensorMetadata(
        name="input",
        shape=(1, 3, 224, 224),
        dtype=TensorProto.FLOAT,
    )
    repr_str = repr(metadata)
    assert "input" in repr_str
    assert "(1, 3, 224, 224)" in repr_str


def test_tensor_metadata_eq():
    """测试张量元数据相等比较"""
    m1 = TensorMetadata(name="x", shape=(2, 3), dtype=TensorProto.FLOAT)
    m2 = TensorMetadata(name="x", shape=(2, 3), dtype=TensorProto.FLOAT)
    m3 = TensorMetadata(name="y", shape=(2, 3), dtype=TensorProto.FLOAT)

    assert m1 == m2
    assert m1 != m3
