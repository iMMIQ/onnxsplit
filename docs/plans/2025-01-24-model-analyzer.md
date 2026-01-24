# Plan 2: ONNX模型分析模块

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现ONNX模型解析、形状推断和依赖关系图构建功能。

**Architecture:**
- 使用 ONNX Python API 解析模型图结构
- 利用 ONNX 内置的 shape inference 推断张量形状
- 构建有向无环图(DAG)表示算子间的数据依赖关系

**Tech Stack:** onnx>=1.20.1, networkx (可选，用于图操作), pytest

---

## Task 1: 张量元数据结构

**Files:**
- Create: `src/onnxsplit/analyzer/tensor.py`
- Test: `tests/test_analyzer_tensor.py`

**Step 1: 编写张量元数据测试**

创建 `tests/test_analyzer_tensor.py`:

```python
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
        dtype=TensorProto.FLOAT32,
    )
    # 1M * 4 bytes = 4MB
    assert metadata.size_mb == pytest.approx(4.0, rel=0.01)


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
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_analyzer_tensor.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.analyzer.tensor`

**Step 3: 实现张量元数据结构**

创建 `src/onnxsplit/analyzer/tensor.py`:

```python
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
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_analyzer_tensor.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/analyzer/tensor.py tests/test_analyzer_tensor.py
git commit -m "feat: add tensor metadata structure"
```

---

## Task 2: 算子信息结构

**Files:**
- Create: `src/onnxsplit/analyzer/operator.py`
- Test: `tests/test_analyzer_operator.py`

**Step 1: 编写算子信息测试**

创建 `tests/test_analyzer_operator.py`:

```python
"""测试算子信息结构"""
from onnx import TensorProto, helper
from onnxsplit.analyzer.tensor import TensorMetadata
from onnxsplit.analyzer.operator import OperatorInfo


def test_operator_info_creation():
    """测试创建算子信息"""
    op = OperatorInfo(
        name="/model/Conv_0",
        op_type="Conv",
        attributes={"kernel_shape": [3, 3], "pads": [1, 1, 1, 1]},
        input_tensors=[
            TensorMetadata("input", shape=(1, 3, 224, 224), dtype=TensorProto.FLOAT),
            TensorMetadata("weight", shape=(64, 3, 3, 3), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(1, 64, 112, 112), dtype=TensorProto.FLOAT),
        ],
    )
    assert op.name == "/model/Conv_0"
    assert op.op_type == "Conv"
    assert op.attributes["kernel_shape"] == [3, 3]
    assert len(op.input_tensors) == 2
    assert len(op.output_tensors) == 1


def test_operator_info_input_memory():
    """测试算子输入内存计算"""
    op = OperatorInfo(
        name="test",
        op_type="Add",
        attributes={},
        input_tensors=[
            TensorMetadata("a", shape=(1000, 1000), dtype=TensorProto.FLOAT),
            TensorMetadata("b", shape=(1000, 1000), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("c", shape=(1000, 1000), dtype=TensorProto.FLOAT),
        ],
    )
    # 两个输入各 4MB
    assert op.input_memory_mb == pytest.approx(8.0, rel=0.01)


def test_operator_info_output_memory():
    """测试算子输出内存计算"""
    op = OperatorInfo(
        name="test",
        op_type="Relu",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(500, 500), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(500, 500), dtype=TensorProto.FLOAT),
        ],
    )
    # 输出 1MB
    assert op.output_memory_mb == pytest.approx(1.0, rel=0.01)


def test_operator_info_total_memory():
    """测试算子总内存计算"""
    op = OperatorInfo(
        name="test",
        op_type="Add",
        attributes={},
        input_tensors=[
            TensorMetadata("a", shape=(1000, 1000), dtype=TensorProto.FLOAT),
            TensorMetadata("b", shape=(1000, 1000), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("c", shape=(1000, 1000), dtype=TensorProto.FLOAT),
        ],
    )
    # 输入8MB + 输出4MB = 12MB
    assert op.total_memory_mb == pytest.approx(12.0, rel=0.01)


def test_operator_info_no_inputs():
    """测试无输入的算子"""
    op = OperatorInfo(
        name="test",
        op_type="Constant",
        attributes={},
        input_tensors=[],
        output_tensors=[
            TensorMetadata("output", shape=(10,), dtype=TensorProto.FLOAT),
        ],
    )
    assert op.input_memory_mb == 0
    assert op.output_memory_mb > 0


def test_operator_info_no_outputs():
    """测试无输出的算子"""
    op = OperatorInfo(
        name="test",
        op_type="Dropout",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(100,), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[],
    )
    assert op.output_memory_mb == 0
    assert op.input_memory_mb > 0


def test_operator_info_repr():
    """测试算子信息字符串表示"""
    op = OperatorInfo(
        name="/model/Conv_0",
        op_type="Conv",
        attributes={},
        input_tensors=[],
        output_tensors=[],
    )
    repr_str = repr(op)
    assert "Conv" in repr_str
    assert "/model/Conv_0" in repr_str


def test_operator_info_attribute_get():
    """测试获取算子属性"""
    op = OperatorInfo(
        name="test",
        op_type="Conv",
        attributes={"kernel_shape": [3, 3], "strides": [1, 1]},
        input_tensors=[],
        output_tensors=[],
    )
    assert op.get_attribute("kernel_shape") == [3, 3]
    assert op.get_attribute("strides") == [1, 1]


def test_operator_info_attribute_get_default():
    """测试获取不存在的属性返回默认值"""
    op = OperatorInfo(
        name="test",
        op_type="Conv",
        attributes={},
        input_tensors=[],
        output_tensors=[],
    )
    assert op.get_attribute("nonexistent", default=5) == 5
    assert op.get_attribute("nonexistent") is None


def test_operator_info_get_input_shape():
    """测试获取输入形状"""
    op = OperatorInfo(
        name="test",
        op_type="Add",
        attributes={},
        input_tensors=[
            TensorMetadata("a", shape=(2, 3, 4), dtype=TensorProto.FLOAT),
            TensorMetadata("b", shape=(2, 3, 4), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[],
    )
    assert op.get_input_shape(0) == (2, 3, 4)
    assert op.get_input_shape(1) == (2, 3, 4)


def test_operator_info_get_input_shape_out_of_bounds():
    """测试获取不存在的输入形状"""
    op = OperatorInfo(
        name="test",
        op_type="Add",
        attributes={},
        input_tensors=[
            TensorMetadata("a", shape=(2, 3), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[],
    )
    assert op.get_input_shape(5) is None
    assert op.get_input_shape(1) is None


def test_operator_info_get_output_shape():
    """测试获取输出形状"""
    op = OperatorInfo(
        name="test",
        op_type="Relu",
        attributes={},
        input_tensors=[],
        output_tensors=[
            TensorMetadata("output", shape=(1, 64, 56, 56), dtype=TensorProto.FLOAT),
        ],
    )
    assert op.get_output_shape(0) == (1, 64, 56, 56)


def test_operator_info_from_node_proto():
    """测试从ONNX NodeProto创建算子信息"""
    node = helper.make_node(
        "Conv",
        inputs=["input", "weight", "bias"],
        outputs=["output"],
        name="conv_0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # 此时没有形状信息，需要后续补充
    op = OperatorInfo.from_node_proto(node)
    assert op.name == "conv_0"
    assert op.op_type == "Conv"
    assert op.attributes["kernel_shape"] == [3, 3]
    assert op.attributes["pads"] == [1, 1, 1, 1]
    assert op.input_names == ["input", "weight", "bias"]
    assert op.output_names == ["output"]


def test_operator_info_with_dynamic_shape():
    """测试处理动态形状（包含-1或None）"""
    op = OperatorInfo(
        name="test",
        op_type="Reshape",
        attributes={},
        input_tensors=[
            # 动态batch
            TensorMetadata("input", shape=(-1, 3, 224, 224), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(-1, 3, 224, 224), dtype=TensorProto.FLOAT),
        ],
    )
    # 动态维度无法计算内存，应该返回0或特殊处理
    assert op.input_memory_mb == 0
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_analyzer_operator.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.analyzer.operator`

**Step 3: 实现算子信息结构**

创建 `src/onnxsplit/analyzer/operator.py`:

```python
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
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_analyzer_operator.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/analyzer/operator.py tests/test_analyzer_operator.py
git commit -m "feat: add operator info structure"
```

---

## Task 3: 模型分析器 - 加载与基本信息提取

**Files:**
- Create: `src/onnxsplit/analyzer/model.py`
- Test: `tests/test_analyzer_model.py`
- Create: `tests/fixtures/models/simple_conv.onnx`

**Step 1: 创建简单的测试ONNX模型**

创建测试辅助文件 `tests/fixtures/models/create_simple_model.py`:

```python
"""创建简单测试模型"""
import onnx
from onnx import helper, TensorProto


def create_simple_conv_model() -> onnx.ModelProto:
    """创建一个简单的卷积模型用于测试"""
    # 输入
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 8, 8]
    )

    # 权重
    weight_const = helper.make_tensor(
        "weight", TensorProto.FLOAT, [2, 3, 3, 3], [0.1] * 54
    )
    weight_node = helper.make_node("Constant", [], ["weight_value"], value=weight_const)

    # 卷积
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight_value"],
        outputs=["conv_output"],
        name="conv_0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # ReLU
    relu_node = helper.make_node(
        "Relu",
        inputs=["conv_output"],
        outputs=["relu_output"],
        name="relu_0",
    )

    # 输出
    output_tensor = helper.make_tensor_value_info(
        "relu_output", TensorProto.FLOAT, [1, 2, 8, 8]
    )

    # 图
    graph = helper.make_graph(
        [weight_node, conv_node, relu_node],
        "simple_conv",
        [input_tensor],
        [output_tensor],
    )

    # 模型
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


def create_matmul_model() -> onnx.ModelProto:
    """创建一个MatMul模型"""
    input_a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 4])
    input_b = helper.make_tensor_value_info("B", TensorProto.FLOAT, [4, 3])

    matmul_node = helper.make_node(
        "MatMul", inputs=["A", "B"], outputs=["C"], name="matmul_0"
    )

    output_c = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [matmul_node], "matmul_model", [input_a, input_b], [output_c]
    )

    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


def create_model_with_branches() -> onnx.ModelProto:
    """创建有分支的模型"""
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 8, 8]
    )

    # 分支1
    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "w1"],
        outputs=["conv1_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # 分支2
    conv2 = helper.make_node(
        "Conv",
        inputs=["input", "w2"],
        outputs=["conv2_out"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    # 合并
    add = helper.make_node(
        "Add", inputs=["conv1_out", "conv2_out"], outputs=["output"]
    )

    w1 = helper.make_tensor("w1", TensorProto.FLOAT, [2, 3, 3, 3], [0.1] * 54)
    w2 = helper.make_tensor("w2", TensorProto.FLOAT, [2, 3, 3, 3], [0.2] * 54)

    const1 = helper.make_node("Constant", [], ["w1_value"], value=w1)
    const2 = helper.make_node("Constant", [], ["w2_value"], value=w2)

    # 修正conv节点的输入
    conv1.input[1] = "w1_value"
    conv2.input[1] = "w2_value"

    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 2, 8, 8]
    )

    graph = helper.make_graph(
        [const1, const2, conv1, conv2, add],
        "branch_model",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)
    return model


if __name__ == "__main__":
    import pathlib

    output_dir = pathlib.Path("tests/fixtures/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    onnx.save(create_simple_conv_model(), output_dir / "simple_conv.onnx")
    onnx.save(create_matmul_model(), output_dir / "simple_matmul.onnx")
    onnx.save(create_model_with_branches(), output_dir / "model_with_branches.onnx")

    print("Test models created successfully!")
```

运行创建模型:

```bash
uv run python tests/fixtures/models/create_simple_model.py
```

**Step 2: 编写模型分析器测试**

创建 `tests/test_analyzer_model.py`:

```python
"""测试模型分析器"""
from pathlib import Path
import onnx
from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo


def test_analyzer_load_model():
    """测试加载ONNX模型"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    assert analyzer.model is not None
    assert analyzer.graph is not None


def test_analyzer_get_input_info():
    """测试获取模型输入信息"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    inputs = analyzer.get_inputs()
    assert len(inputs) == 1
    assert inputs[0].name == "input"
    assert inputs[0].shape == (1, 3, 8, 8)


def test_analyzer_get_output_info():
    """测试获取模型输出信息"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    outputs = analyzer.get_outputs()
    assert len(outputs) == 1
    assert outputs[0].name == "relu_output"


def test_analyzer_get_operators():
    """测试获取所有算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    ops = analyzer.get_operators()
    assert len(ops) == 2  # Conv + Relu (Constant被跳过)

    op_types = [op.op_type for op in ops]
    assert "Conv" in op_types
    assert "Relu" in op_types


def test_analyzer_get_operator_by_name():
    """测试按名称获取算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    conv_op = analyzer.get_operator("conv_0")
    assert conv_op is not None
    assert conv_op.op_type == "Conv"
    assert conv_op.name == "conv_0"


def test_analyzer_get_nonexistent_operator():
    """测试获取不存在的算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    op = analyzer.get_operator("nonexistent")
    assert op is None


def test_analyzer_tensor_shapes():
    """测试张量形状推断"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    conv_op = analyzer.get_operator("conv_0")
    assert conv_op is not None
    assert len(conv_op.input_tensors) > 0
    assert len(conv_op.output_tensors) > 0


def test_analyzer_matmul_model():
    """测试分析MatMul模型"""
    model_path = Path("tests/fixtures/models/simple_matmul.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    ops = analyzer.get_operators()
    assert len(ops) == 1
    assert ops[0].op_type == "MatMul"


def test_analyzer_branch_model():
    """测试分析有分支的模型"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    ops = analyzer.get_operators()
    assert len(ops) == 3  # 2 Conv + 1 Add

    # 检查输入被两个Conv使用
    conv_ops = [op for op in ops if op.op_type == "Conv"]
    assert len(conv_ops) == 2


def test_analyzer_model_ir_version():
    """测试获取模型IR版本"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    assert analyzer.ir_version > 0
    assert analyzer.opset_version > 0


def test_analyzer_producer_info():
    """测试获取生产者信息"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    # 生产者信息可能为空，但不应该报错
    assert hasattr(analyzer, "producer_name")
    assert hasattr(analyzer, "producer_version")


def test_analyzer_from_model_proto():
    """测试从ModelProto创建分析器"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    model = onnx.load(model_path)

    analyzer = ModelAnalyzer.from_model_proto(model)
    assert analyzer.model is not None

    ops = analyzer.get_operators()
    assert len(ops) >= 1


def test_analyzer_constant_skipped():
    """测试Constant算子被正确处理"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    # Constant算子通常不作为独立算子分析
    ops = analyzer.get_operators()
    for op in ops:
        assert op.op_type != "Constant"
```

**Step 3: 运行测试验证失败**

Run: `uv run pytest tests/test_analyzer_model.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.analyzer.model`

**Step 4: 实现模型分析器**

创建 `src/onnxsplit/analyzer/model.py`:

```python
"""ONNX模型分析器"""
from pathlib import Path
from typing import Optional

import onnx
from onnx import ModelProto, GraphProto, ValueInfoProto, NodeProto

from onnxsplit.analyzer.tensor import TensorMetadata, dtype_to_bytes
from onnxsplit.analyzer.operator import OperatorInfo


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
                    op_info.input_tensors.append(
                        TensorMetadata(input_name, shape, dtype)
                    )

            # 添加输出张量信息
            for output_name in node.output:
                shape = self._get_tensor_shape(output_name)
                dtype = self._get_tensor_dtype(output_name)
                if shape:
                    op_info.output_tensors.append(
                        TensorMetadata(output_name, shape, dtype)
                    )

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
                return node.name or f"{node.op_type}_{node.output[0]}"
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
                name = node.name or f"{node.op_type}_{node.output[0]}"
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
```

**Step 5: 运行测试验证通过**

Run: `uv run pytest tests/test_analyzer_model.py -v`
Expected: PASS - 所有测试通过

**Step 6: 提交**

```bash
git add src/onnxsplit/analyzer/model.py tests/test_analyzer_model.py tests/fixtures/models
git commit -m "feat: add model analyzer"
```

---

## Task 4: 依赖关系图构建

**Files:**
- Create: `src/onnxsplit/analyzer/dependency.py`
- Test: `tests/test_analyzer_dependency.py`

**Step 1: 编写依赖关系图测试**

创建 `tests/test_analyzer_dependency.py`:

```python
"""测试依赖关系图构建"""
from onnxsplit.analyzer.dependency import DependencyGraph, DependencyEdge
from onnxsplit.analyzer.model import ModelAnalyzer
from pathlib import Path


def test_dependency_graph_creation():
    """测试创建依赖关系图"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    assert dep_graph is not None
    assert len(dep_graph.nodes) > 0


def test_dependency_graph_nodes():
    """测试依赖图节点"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    # 检查算子节点存在
    node_names = list(dep_graph.nodes.keys())
    assert "conv_0" in node_names


def test_dependency_graph_edges():
    """测试依赖图边"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    # conv_0 -> relu_0
    edges = dep_graph.get_outgoing_edges("conv_0")
    assert len(edges) > 0
    assert any(e.dst == "relu_0" for e in edges)


def test_dependency_graph_incoming_edges():
    """测试获取入边"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    edges = dep_graph.get_incoming_edges("relu_0")
    assert len(edges) > 0
    assert any(e.src == "conv_0" for e in edges)


def test_dependency_graph_topological_order():
    """测试拓扑排序"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    order = dep_graph.topological_sort()
    assert len(order) > 0
    # conv_0 应该在 relu_0 之前
    if "conv_0" in order and "relu_0" in order:
        assert order.index("conv_0") < order.index("relu_0")


def test_dependency_graph_branch_model():
    """测试分支模型的依赖图"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    # 应该有两个Conv分支
    conv_nodes = [n for n in dep_graph.nodes.values() if n.op_type == "Conv"]
    assert len(conv_nodes) == 2

    # 两个Conv都应该连接到Add
    add_node = next((n for n in dep_graph.nodes.values() if n.op_type == "Add"), None)
    assert add_node is not None

    incoming = dep_graph.get_incoming_edges(add_node.name)
    assert len(incoming) == 2


def test_dependency_graph_get_predecessors():
    """测试获取前驱节点"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    predecessors = dep_graph.get_predecessors("relu_0")
    assert "conv_0" in predecessors


def test_dependency_graph_get_successors():
    """测试获取后继节点"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    successors = dep_graph.get_successors("conv_0")
    assert "relu_0" in successors


def test_dependency_graph_source_nodes():
    """测试获取源节点（无入边的节点）"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    sources = dep_graph.get_source_nodes()
    # Conv节点应该是源节点（输入是图输入，不是其他算子的输出）
    conv_nodes = [n for n in sources if n.op_type == "Conv"]
    assert len(conv_nodes) == 2


def test_dependency_graph_sink_nodes():
    """测试获取汇节点（无出边的节点）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    sinks = dep_graph.get_sink_nodes()
    # 最后的Relu应该是汇节点
    relu_nodes = [n for n in sinks if n.op_type == "Relu"]
    assert len(relu_nodes) == 1


def test_dependency_edge_repr():
    """测试依赖边的字符串表示"""
    edge = DependencyEdge(src="A", dst="B", tensor_name="data")
    repr_str = repr(edge)
    assert "A" in repr_str
    assert "B" in repr_str


def test_dependency_graph_has_path():
    """测试路径检查"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    assert dep_graph.has_path("conv_0", "relu_0")
    assert not dep_graph.has_path("relu_0", "conv_0")


def test_dependency_graph_cycles():
    """测试循环检测（简单模型应该无环）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    assert not dep_graph.has_cycle()
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_analyzer_dependency.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.analyzer.dependency`

**Step 3: 实现依赖关系图**

创建 `src/onnxsplit/analyzer/dependency.py`:

```python
"""依赖关系图构建"""
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict, deque

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo


@dataclass(frozen=True)
class DependencyEdge:
    """依赖边

    Attributes:
        src: 源算子名称
        dst: 目标算子名称
        tensor_name: 传递的张量名称
    """
    src: str
    dst: str
    tensor_name: str


class DependencyNode:
    """依赖图节点

    Attributes:
        name: 算子名称
        op_type: 算子类型
        info: 算子信息
    """

    def __init__(self, name: str, op_type: str, info: OperatorInfo):
        self.name = name
        self.op_type = op_type
        self.info = info

    def __repr__(self) -> str:
        return f"DependencyNode(name={self.name!r}, op_type={self.op_type!r})"


class DependencyGraph:
    """算子依赖关系图

    有向图，表示算子间的数据依赖关系。
    边 A -> B 表示 A 的输出是 B 的输入。
    """

    def __init__(self):
        """初始化空的依赖图"""
        self.nodes: dict[str, DependencyNode] = {}
        self._outgoing: dict[str, list[DependencyEdge]] = defaultdict(list)
        self._incoming: dict[str, list[DependencyEdge]] = defaultdict(list)

    @classmethod
    def build(cls, analyzer: ModelAnalyzer) -> "DependencyGraph":
        """从模型分析器构建依赖图

        Args:
            analyzer: 模型分析器

        Returns:
            DependencyGraph实例
        """
        graph = cls()

        # 添加所有算子作为节点
        for op_info in analyzer.get_operators():
            node = DependencyNode(op_info.name, op_info.op_type, op_info)
            graph.add_node(node)

        # 构建边（基于数据流）
        for op_info in analyzer.get_operators():
            for input_name in op_info.input_names:
                # 找到产生这个输入的算子
                producer = analyzer.get_tensor_producer(input_name)
                if producer and producer in graph.nodes:
                    # 创建边: producer -> current_op
                    edge = DependencyEdge(producer, op_info.name, input_name)
                    graph.add_edge(edge)

        return graph

    def add_node(self, node: DependencyNode) -> None:
        """添加节点

        Args:
            node: 依赖图节点
        """
        self.nodes[node.name] = node

    def add_edge(self, edge: DependencyEdge) -> None:
        """添加边

        Args:
            edge: 依赖边
        """
        self._outgoing[edge.src].append(edge)
        self._incoming[edge.dst].append(edge)

    def get_outgoing_edges(self, node_name: str) -> list[DependencyEdge]:
        """获取节点的出边

        Args:
            node_name: 节点名称

        Returns:
            出边列表
        """
        return self._outgoing.get(node_name, [])

    def get_incoming_edges(self, node_name: str) -> list[DependencyEdge]:
        """获取节点的入边

        Args:
            node_name: 节点名称

        Returns:
            入边列表
        """
        return self._incoming.get(node_name, [])

    def get_predecessors(self, node_name: str) -> set[str]:
        """获取前驱节点集合

        Args:
            node_name: 节点名称

        Returns:
            前驱节点名称集合
        """
        return {edge.src for edge in self._incoming.get(node_name, [])}

    def get_successors(self, node_name: str) -> set[str]:
        """获取后继节点集合

        Args:
            node_name: 节点名称

        Returns:
            后继节点名称集合
        """
        return {edge.dst for edge in self._outgoing.get(node_name, [])}

    def get_source_nodes(self) -> list[DependencyNode]:
        """获取源节点（无入边的节点）

        Returns:
            源节点列表
        """
        return [
            node for name, node in self.nodes.items()
            if not self._incoming.get(name)
        ]

    def get_sink_nodes(self) -> list[DependencyNode]:
        """获取汇节点（无出边的节点）

        Returns:
            汇节点列表
        """
        return [
            node for name, node in self.nodes.items()
            if not self._outgoing.get(name)
        ]

    def topological_sort(self) -> list[str]:
        """执行拓扑排序

        Returns:
            拓扑排序后的节点名称列表

        Raises:
            ValueError: 图中存在环
        """
        # Kahn算法
        in_degree = {name: 0 for name in self.nodes}
        for edges in self._incoming.values():
            for edge in edges:
                in_degree[edge.dst] += 1

        # 找出所有入度为0的节点
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # 减少后继节点的入度
            for edge in self._outgoing.get(node, []):
                in_degree[edge.dst] -= 1
                if in_degree[edge.dst] == 0:
                    queue.append(edge.dst)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def has_path(self, src: str, dst: str) -> bool:
        """检查是否存在从src到dst的路径

        Args:
            src: 源节点名称
            dst: 目标节点名称

        Returns:
            存在路径返回True
        """
        if src == dst:
            return True

        visited = set()
        queue = deque([src])

        while queue:
            node = queue.popleft()
            if node == dst:
                return True

            if node in visited:
                continue
            visited.add(node)

            for edge in self._outgoing.get(node, []):
                if edge.dst not in visited:
                    queue.append(edge.dst)

        return False

    def has_cycle(self) -> bool:
        """检测图中是否存在环

        Returns:
            存在环返回True
        """
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def __repr__(self) -> str:
        return (
            f"DependencyGraph(nodes={len(self.nodes)}, "
            f"edges={sum(len(e) for e in self._outgoing.values())})"
        )
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_analyzer_dependency.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/analyzer/dependency.py tests/test_analyzer_dependency.py
git commit -m "feat: add dependency graph builder"
```

---

## Task 5: 分析器模块导出

**Files:**
- Modify: `src/onnxsplit/analyzer/__init__.py`

**Step 1: 导出分析器模块公共接口**

编辑 `src/onnxsplit/analyzer/__init__.py`:

```python
"""ONNX模型分析模块

提供ONNX模型解析、形状推断和依赖关系图构建功能。
"""

from onnxsplit.analyzer.tensor import TensorMetadata, dtype_to_bytes
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.dependency import (
    DependencyGraph,
    DependencyNode,
    DependencyEdge,
)


__all__ = [
    # Tensor
    "TensorMetadata",
    "dtype_to_bytes",
    # Operator
    "OperatorInfo",
    # Model
    "ModelAnalyzer",
    # Dependency
    "DependencyGraph",
    "DependencyNode",
    "DependencyEdge",
]
```

**Step 2: 验证模块导入**

Run: `uv run python -c "from onnxsplit.analyzer import *; print('Import successful')"`
Expected: 打印 "Import successful"

**Step 3: 运行所有分析器测试**

Run: `uv run pytest tests/test_analyzer_*.py -v`
Expected: PASS - 所有测试通过

**Step 4: 提交**

```bash
git add src/onnxsplit/analyzer/__init__.py
git commit -m "chore: export analyzer module public API"
```

---

## 完成检查

**Step 1: 运行所有测试**

Run: `uv run pytest tests/ -v`
Expected: PASS - 所有测试通过

**Step 2: 检查代码风格**

Run: `uv run ruff check src/onnxsplit/analyzer tests/`
Expected: 无错误

**Step 3: 检查测试覆盖率**

Run: `uv run pytest tests/test_analyzer_*.py --cov=onnxsplit/analyzer --cov-report=term-missing`
Expected: 覆盖率 >= 85%

**Step 4: 最终提交**

```bash
git add .
git commit -m "chore: finalize plan 2 - ONNX model analyzer module"
```

---

**Plan 2 完成！** ONNX模型分析模块已实现，包括：
- 张量元数据结构
- 算子信息结构
- 模型加载和解析
- 依赖关系图构建
- 形状推断
- 完整的测试覆盖
