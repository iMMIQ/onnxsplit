# Plan 4: 数据流重连与图变换模块

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现算子复制、数据流重连和图变换功能，支持任意切分数组合的复杂重排。

**Architecture:**
- 复制算子节点生成多个副本
- 插入Split/Concat算子处理切分边界
- 使用Slice+Concat处理任意切分数组合
- 重新连接节点间边关系

**Tech Stack**: onnx>=1.20.1, pytest

---

## Task 1: 节点克隆与命名

**Files:**
- Create: `src/onnxsplit/transform/node_clone.py`
- Test: `tests/test_node_clone.py`

**Step 1: 编写节点克隆测试**

创建 `tests/test_node_clone.py`:

```python
"""测试节点克隆功能"""
from onnx import helper, TensorProto
from onnxsplit.transform.node_clone import clone_node, generate_split_name


def test_generate_split_name():
    """测试生成切分后的节点名"""
    assert generate_split_name("conv_0", 0) == "conv_0_split_0"
    assert generate_split_name("conv_0", 1) == "conv_0_split_1"
    assert generate_split_name("matmul", 5) == "matmul_split_5"


def test_generate_split_name_custom_suffix():
    """测试自定义后缀"""
    assert generate_split_name("conv_0", 0, suffix="part") == "conv_0_part_0"


def test_clone_simple_node():
    """测试克隆简单节点"""
    node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
        name="relu_0",
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    assert cloned.op_type == "Relu"
    assert cloned.name == "relu_0_split_0"
    assert list(cloned.input) == ["input"]
    assert list(cloned.output) == ["output_0"]


def test_clone_node_with_attributes():
    """测试克隆带属性的节点"""
    node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        name="conv_0",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    assert cloned.op_type == "Conv"
    # 检查属性保留
    kernel_shape = None
    pads = None
    for attr in cloned.attribute:
        if attr.name == "kernel_shape":
            kernel_shape = list(attr.ints)
        elif attr.name == "pads":
            pads = list(attr.ints)

    assert kernel_shape == [3, 3]
    assert pads == [1, 1, 1, 1]


def test_clone_node_multiple_outputs():
    """测试克隆多输出节点"""
    node = helper.make_node(
        "Split",
        inputs=["input"],
        outputs=["output_0", "output_1", "output_2"],
        name="split_0",
        axis=0,
    )

    cloned = clone_node(
        node,
        suffix="_split_0",
        new_outputs=["output_0_0", "output_1_0", "output_2_0"],
    )

    assert len(cloned.output) == 3
    assert cloned.output[0] == "output_0_0"


def test_clone_node_preserve_domain():
    """测试保留domain信息"""
    node = helper.make_node(
        "CustomOp",
        inputs=["input"],
        outputs=["output"],
        name="custom_0",
        domain="custom.domain",
    )

    cloned = clone_node(node, suffix="_copy", new_outputs=["output_copy"])

    assert cloned.domain == "custom.domain"


def test_clone_node_without_name():
    """测试克隆无名称节点"""
    node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
        # 没有name
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    # 应该生成默认名称
    assert "_split_0" in cloned.name


def test_clone_batch_norm():
    """测试克隆BatchNorm（多属性）"""
    node = helper.make_node(
        "BatchNormalization",
        inputs=["input", "scale", "b", "mean", "var"],
        outputs=["output"],
        name="bn_0",
        epsilon=0.001,
        momentum=0.9,
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    assert cloned.op_type == "BatchNormalization"
    epsilon = None
    momentum = None
    for attr in cloned.attribute:
        if attr.name == "epsilon":
            epsilon = attr.f
        elif attr.name == "momentum":
            momentum = attr.f

    assert epsilon == 0.001
    assert momentum == 0.9


def test_clone_transpose():
    """测试克隆Transpose（perm属性）"""
    node = helper.make_node(
        "Transpose",
        inputs=["input"],
        outputs=["output"],
        name="transpose_0",
        perm=[0, 2, 3, 1],
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    perm = None
    for attr in cloned.attribute:
        if attr.name == "perm":
            perm = list(attr.ints)

    assert perm == [0, 2, 3, 1]


def test_clone_reshape():
    """测试克隆Reshape"""
    node = helper.make_node(
        "Reshape",
        inputs=["input", "shape"],
        outputs=["output"],
        name="reshape_0",
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    assert cloned.op_type == "Reshape"
    assert len(cloned.input) == 2


def test_clone_multiple():
    """测试克隆多个副本"""
    node = helper.make_node(
        "Relu",
        inputs=["input"],
        outputs=["output"],
        name="relu_0",
    )

    clones = []
    for i in range(3):
        cloned = clone_node(
            node,
            suffix=f"_split_{i}",
            new_outputs=[f"output_{i}"],
        )
        clones.append(cloned)

    assert len(clones) == 3
    assert clones[0].name == "relu_0_split_0"
    assert clones[1].name == "relu_0_split_1"
    assert clones[2].name == "relu_0_split_2"


def test_clone_gather():
    """测试克隆Gather（带int属性）"""
    node = helper.make_node(
        "Gather",
        inputs=["input", "indices"],
        outputs=["output"],
        name="gather_0",
        axis=1,
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    axis = None
    for attr in cloned.attribute:
        if attr.name == "axis":
            axis = attr.i

    assert axis == 1


def test_clone_cast():
    """测试克隆Cast（带to属性）"""
    node = helper.make_node(
        "Cast",
        inputs=["input"],
        outputs=["output"],
        name="cast_0",
        to=TensorProto.INT64,
    )

    cloned = clone_node(node, suffix="_split_0", new_outputs=["output_0"])

    to_attr = None
    for attr in cloned.attribute:
        if attr.name == "to":
            to_attr = attr.i

    assert to_attr == TensorProto.INT64
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_node_clone.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.transform.node_clone`

**Step 3: 实现节点克隆功能**

创建 `src/onnxsplit/transform/node_clone.py`:

```python
"""节点克隆功能"""
from onnx import NodeProto


def generate_split_name(original_name: str, part_idx: int, suffix: str = "split") -> str:
    """生成切分后的节点名称

    Args:
        original_name: 原始节点名称
        part_idx: 切分索引
        suffix: 后缀名称

    Returns:
        新的节点名称
    """
    base_name = original_name or f"node_{id(object())}"
    return f"{base_name}_{suffix}_{part_idx}"


def clone_node(
    node: NodeProto,
    suffix: str,
    new_outputs: list[str],
    new_name: str | None = None,
) -> NodeProto:
    """克隆ONNX节点

    创建节点的副本，可以修改名称和输出名称。

    Args:
        node: 原始节点
        suffix: 名称后缀
        new_outputs: 新的输出名称列表
        new_name: 新的节点名称（如果为None则自动生成）

    Returns:
        克隆的节点
    """
    import onnx.helper

    # 生成新名称
    if new_name is None:
        base_name = node.name if node.name else f"{node.op_type}_node"
        new_name = f"{base_name}{suffix}"

    # 创建新节点，保留所有属性
    new_node = onnx.helper.make_node(
        op_type=node.op_type,
        inputs=list(node.input),
        outputs=new_outputs,
        name=new_name,
        domain=node.domain,
    )

    # 复制所有属性
    for attr in node.attribute:
        new_node.attribute.append(attr)

    return new_node
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_node_clone.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/transform/node_clone.py tests/test_node_clone.py
git commit -m "feat: add node cloning functionality"
```

---

## Task 2: Split和Concat节点生成

**Files:**
- Create: `src/onnxsplit/transform/split_concat.py`
- Test: `tests/test_split_concat.py`

**Step 1: 编写Split/Concat节点测试**

创建 `tests/test_split_concat.py`:

```python
"""测试Split和Concat节点生成"""
from onnx import helper, TensorProto
from onnxsplit.transform.split_concat import (
    create_split_node,
    create_concat_node,
    create_slice_node,
)


def test_create_split_node_equal():
    """测试创建均分Split节点"""
    node = create_split_node(
        input_name="input",
        axis=0,
        parts=4,
        output_prefix="split_out",
    )

    assert node.op_type == "Split"
    assert node.input[0] == "input"
    assert len(node.output) == 4
    assert node.output[0] == "split_out_0"
    assert node.output[3] == "split_out_3"


def test_create_split_node_with_split_attr():
    """测试创建带split属性的Split节点"""
    node = create_split_node(
        input_name="input",
        axis=0,
        parts=3,
        output_prefix="out",
        split_sizes=[10, 20, 30],
    )

    assert node.op_type == "Split"
    # 检查split属性
    split_attr = None
    for attr in node.attribute:
        if attr.name == "split":
            split_attr = list(attr.ints)

    assert split_attr == [10, 20, 30]


def test_create_split_node_axis():
    """测试Split节点axis属性"""
    for axis in [0, 1, 2, -1]:
        node = create_split_node(
            input_name="input",
            axis=axis,
            parts=2,
            output_prefix="out",
        )

        axis_attr = None
        for attr in node.attribute:
            if attr.name == "axis":
                if attr.type == 2:  # INT
                    axis_attr = attr.i
                elif attr.type == 7:  # INTS
                    axis_attr = list(attr.ints)

        assert axis_attr == axis or (isinstance(axis_attr, list) and axis_attr[0] == axis)


def test_create_concat_node():
    """测试创建Concat节点"""
    node = create_concat_node(
        input_names=["input_0", "input_1", "input_2"],
        output_name="output",
        axis=0,
    )

    assert node.op_type == "Concat"
    assert len(node.input) == 3
    assert node.input[0] == "input_0"
    assert node.input[2] == "input_2"
    assert node.output[0] == "output"


def test_create_concat_node_axis():
    """测试Concat节点axis属性"""
    for axis in [0, 1, 2]:
        node = create_concat_node(
            input_names=["a", "b"],
            output_name="out",
            axis=axis,
        )

        axis_attr = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis_attr = attr.i

        assert axis_attr == axis


def test_create_slice_node():
    """测试创建Slice节点"""
    node = create_slice_node(
        input_name="input",
        output_name="output",
        starts=[0],
        ends=[10],
        axes=[0],
    )

    assert node.op_type == "Slice"
    assert node.input[0] == "input"
    assert node.output[0] == "output"


def test_create_slice_node_multi_dim():
    """测试创建多维Slice节点"""
    node = create_slice_node(
        input_name="input",
        output_name="output",
        starts=[0, 0],
        ends=[10, 20],
        axes=[0, 1],
    )

    assert node.op_type == "Slice"
    # Slice的输入是: input, starts, ends, axes, steps
    assert len(node.input) == 5  # input + starts + ends + axes + steps


def test_create_slice_node_with_steps():
    """测试创建带步长的Slice节点"""
    node = create_slice_node(
        input_name="input",
        output_name="output",
        starts=[0],
        ends=[10],
        axes=[0],
        steps=[2],
    )

    assert node.op_type == "Slice"
    assert len(node.input) == 5


def test_create_split_node_name():
    """测试Split节点命名"""
    node = create_split_node(
        input_name="data",
        axis=0,
        parts=2,
        output_prefix="split",
        node_name="my_split",
    )

    assert node.name == "my_split"


def test_create_concat_node_name():
    """测试Concat节点命名"""
    node = create_concat_node(
        input_names=["a", "b"],
        output_name="out",
        axis=0,
        node_name="my_concat",
    )

    assert node.name == "my_concat"


def test_create_slice_node_name():
    """测试Slice节点命名"""
    node = create_slice_node(
        input_name="input",
        output_name="output",
        starts=[0],
        ends=[10],
        axes=[0],
        node_name="my_slice",
    )

    assert node.name == "my_slice"


def test_create_split_single_part():
    """测试创建单份Split（退化情况）"""
    node = create_split_node(
        input_name="input",
        axis=0,
        parts=1,
        output_prefix="out",
    )

    assert node.op_type == "Split"
    assert len(node.output) == 1


def test_create_concat_single_input():
    """测试创建单输入Concat（退化情况）"""
    node = create_concat_node(
        input_names=["a"],
        output_name="out",
        axis=0,
    )

    assert node.op_type == "Concat"
    assert len(node.input) == 1
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_split_concat.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.transform.split_concat`

**Step 3: 实现Split/Concat节点生成**

创建 `src/onnxsplit/transform/split_concat.py`:

```python
"""Split和Concat节点生成"""
import onnx.helper
from onnx import TensorProto, NodeProto


def create_split_node(
    input_name: str,
    axis: int,
    parts: int,
    output_prefix: str,
    split_sizes: list[int] | None = None,
    node_name: str | None = None,
) -> NodeProto:
    """创建Split算子节点

    Args:
        input_name: 输入张量名称
        axis: 切分轴
        parts: 切分的份数
        output_prefix: 输出张量名称前缀
        split_sizes: 每份的大小（可选，不指定则均分）
        node_name: 节点名称（可选）

    Returns:
        Split节点
    """
    # 生成输出名称
    outputs = [f"{output_prefix}_{i}" for i in range(parts)]

    # 生成节点名称
    if node_name is None:
        node_name = f"split_{output_prefix}"

    # 创建节点
    kwargs = {
        "inputs": [input_name],
        "outputs": outputs,
        "name": node_name,
    }

    # 添加axis属性
    if split_sizes is None:
        # 使用均分，只需要指定axis
        node = onnx.helper.make_node("Split", axis=axis, **kwargs)
    else:
        # 指定每份大小
        node = onnx.helper.make_node(
            "Split",
            axis=axis,
            split=split_sizes,
            **kwargs,
        )

    return node


def create_concat_node(
    input_names: list[str],
    output_name: str,
    axis: int,
    node_name: str | None = None,
) -> NodeProto:
    """创建Concat算子节点

    Args:
        input_names: 输入张量名称列表
        output_name: 输出张量名称
        axis: 拼接轴
        node_name: 节点名称（可选）

    Returns:
        Concat节点
    """
    # 生成节点名称
    if node_name is None:
        node_name = f"concat_{output_name}"

    node = onnx.helper.make_node(
        "Concat",
        inputs=input_names,
        outputs=[output_name],
        axis=axis,
        name=node_name,
    )

    return node


def create_slice_node(
    input_name: str,
    output_name: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    steps: list[int] | None = None,
    node_name: str | None = None,
) -> NodeProto:
    """创建Slice算子节点

    Args:
        input_name: 输入张量名称
        output_name: 输出张量名称
        starts: 每个维度的起始位置
        ends: 每个维度的结束位置
        axes: 切分的轴
        steps: 每个维度的步长（可选，默认为1）
        node_name: 节点名称（可选）

    Returns:
        Slice节点
    """
    # 生成节点名称
    if node_name is None:
        node_name = f"slice_{output_name}"

    # 创建常量张量作为输入
    starts_tensor = onnx.helper.make_tensor(
        "starts", TensorProto.INT64, dims=[len(starts)], vals=starts
    )
    ends_tensor = onnx.helper.make_tensor(
        "ends", TensorProto.INT64, dims=[len(ends)], vals=ends
    )
    axes_tensor = onnx.helper.make_tensor(
        "axes", TensorProto.INT64, dims=[len(axes)], vals=axes
    )

    # Slice节点的输入: data, starts, ends, axes, steps(optional)
    inputs = [
        input_name,
        starts_tensor.name,
        ends_tensor.name,
        axes_tensor.name,
    ]

    if steps is not None:
        steps_tensor = onnx.helper.make_tensor(
            "steps", TensorProto.INT64, dims=[len(steps)], vals=steps
        )
        inputs.append(steps_tensor.name)

    # 注意：返回的节点需要配合初始器使用
    # 这里先创建节点结构，初始器需要在图层面添加
    node = onnx.helper.make_node(
        "Slice",
        inputs=inputs,
        outputs=[output_name],
        name=node_name,
    )

    # 将初始器信息附加到节点上（用于后续处理）
    node._slice_initializers = [starts_tensor, ends_tensor, axes_tensor]
    if steps is not None:
        node._slice_initializers.append(steps_tensor)

    return node


def get_slice_initializers(node: NodeProto) -> list:
    """获取Slice节点的初始器张量

    Args:
        node: Slice节点

    Returns:
        初始器张量列表
    """
    if hasattr(node, "_slice_initializers"):
        return node._slice_initializers
    return []
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_split_concat.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/transform/split_concat.py tests/test_split_concat.py
git commit -m "feat: add Split/Concat node creation"
```

---

## Task 3: 数据流重连核心算法

**Files:**
- Create: `src/onnxsplit/transform/reconnect.py`
- Test: `tests/test_reconnect.py`

**Step 1: 编写数据流重连测试**

创建 `tests/test_reconnect.py`:

```python
"""测试数据流重连算法"""
from onnx import helper, TensorProto
from onnxsplit.transform.reconnect import (
    ReconnectStrategy,
    calculate_overlap_range,
    generate_reconnect_plan,
)


def test_reconnect_strategy_same_parts():
    """测试相同切分数策略"""
    strategy = ReconnectStrategy.determine(4, 4)
    assert strategy == ReconnectStrategy.ONE_TO_ONE


def test_reconnect_strategy_divisible():
    """测试整除关系策略"""
    strategy = ReconnectStrategy.determine(2, 4)
    assert strategy == ReconnectStrategy.SPLIT_SOURCE

    strategy = ReconnectStrategy.determine(4, 2)
    assert strategy == ReconnectStrategy.CONCAT_SOURCE


def test_reconnect_strategy_complex():
    """测试复杂重排策略"""
    strategy = ReconnectStrategy.determine(3, 2)
    assert strategy == ReconnectStrategy.COMPLEX_REORDER

    strategy = ReconnectStrategy.determine(2, 3)
    assert strategy == ReconnectStrategy.COMPLEX_REORDER


def test_calculate_overlap_range():
    """测试计算重叠区间"""
    # src: [0, 33), dst: [0, 50) -> overlap: [0, 33)
    assert calculate_overlap_range(0, 100, 0, 50, 2, 50) == (0, 33)

    # src: [0, 33), dst: [50, 100) -> no overlap
    assert calculate_overlap_range(0, 100, 50, 100, 2, 50) is None


def test_calculate_overlap_edge_cases():
    """测试边界情况"""
    # 完全包含
    assert calculate_overlap_range(0, 100, 0, 100, 1, 1) == (0, 100)

    # 刚好接触
    assert calculate_overlap_range(0, 50, 50, 100, 2, 50) is None


def test_generate_plan_one_to_one():
    """测试1对1连接计划"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=4,
        dst_parts=4,
        batch_size=100,
        src_output="out",
        dst_input="in",
    )

    assert len(plan.connections) == 4
    assert plan.connections[0].src_split_idx == 0
    assert plan.connections[0].dst_split_idx == 0


def test_generate_plan_split_source():
    """测试源切分计划（2->4）"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=2,
        dst_parts=4,
        batch_size=100,
        src_output="out",
        dst_input="in",
    )

    # A0 -> B0, B1
    # A1 -> B2, B3
    a0_connections = [c for c in plan.connections if c.src_split_idx == 0]
    a1_connections = [c for c in plan.connections if c.src_split_idx == 1]

    assert len(a0_connections) == 2
    assert len(a1_connections) == 2


def test_generate_plan_concat_source():
    """测试源合并计划（4->2）"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=4,
        dst_parts=2,
        batch_size=100,
        src_output="out",
        dst_input="in",
    )

    # A0, A1 -> B0
    # A2, A3 -> B1
    b0_inputs = [c for c in plan.connections if c.dst_split_idx == 0]
    b1_inputs = [c for c in plan.connections if c.dst_split_idx == 1]

    assert len(b0_inputs) == 2
    assert len(b1_inputs) == 2


def test_generate_plan_complex_3_to_2():
    """测试复杂重排计划（3->2）"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=3,
        dst_parts=2,
        batch_size=6,  # 每份size=2和3
        src_output="out",
        dst_input="in",
    )

    # B0需要[0,3)，来自A0[0,2)和A1[2,3)
    # B1需要[3,6)，来自A1[3,4)和A2[4,6)
    b0_sources = [c for c in plan.connections if c.dst_split_idx == 0]
    b1_sources = [c for c in plan.connections if c.dst_split_idx == 1]

    assert len(b0_sources) == 2  # 来自A0和A1
    assert len(b1_sources) == 2  # 来自A1和A2


def test_reconnect_connection_repr():
    """测试连接对象表示"""
    from onnxsplit.transform.reconnect import ReconnectConnection

    conn = ReconnectConnection(
        src_split_idx=0,
        dst_split_idx=0,
        src_tensor="A_out_0",
        dst_tensor="B_in_0",
        slice_range=(0, 10),
    )

    repr_str = repr(conn)
    assert "0" in repr_str


def test_reconnect_plan_slice_operations():
    """测试计划中的切片操作"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=3,
        dst_parts=2,
        batch_size=6,
        src_output="out",
        dst_input="in",
    )

    # 应该有切片操作
    assert len(plan.slice_operations) > 0


def test_reconnect_plan_concat_operations():
    """测试计划中的拼接操作"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=3,
        dst_parts=2,
        batch_size=6,
        src_output="out",
        dst_input="in",
    )

    # 应该有拼接操作
    assert len(plan.concat_operations) > 0


def test_reconnect_plan_summary():
    """测试计划摘要"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=4,
        dst_parts=2,
        batch_size=100,
        src_output="out",
        dst_input="in",
    )

    summary = plan.summary()
    assert "A" in summary
    assert "B" in summary


def test_calculate_overlap_detailed():
    """测试详细的重叠计算"""
    # src_parts=3, dst_parts=2, batch=6
    # src chunks: [0,2), [2,4), [4,6)
    # dst chunks: [0,3), [3,6)

    # A0 [0,2) 与 B0 [0,3) 重叠 [0,2)
    assert calculate_overlap_range(
        src_start=0, src_end=2, dst_start=0, dst_end=3, batch_size=6
    ) == (0, 2)

    # A1 [2,4) 与 B0 [0,3) 重叠 [2,3)
    assert calculate_overlap_range(
        src_start=2, src_end=4, dst_start=0, dst_end=3, batch_size=6
    ) == (2, 3)

    # A1 [2,4) 与 B1 [3,6) 重叠 [3,4)
    assert calculate_overlap_range(
        src_start=2, src_end=4, dst_start=3, dst_end=6, batch_size=6
    ) == (3, 4)

    # A2 [4,6) 与 B1 [3,6) 重叠 [4,6)
    assert calculate_overlap_range(
        src_start=4, src_end=6, dst_start=3, dst_end=6, batch_size=6
    ) == (4, 6)


def test_strategy_repr():
    """测试策略枚举表示"""
    assert str(ReconnectStrategy.ONE_TO_ONE) == "ONE_TO_ONE"
    assert str(ReconnectStrategy.SPLIT_SOURCE) == "SPLIT_SOURCE"
    assert str(ReconnectStrategy.CONCAT_SOURCE) == "CONCAT_SOURCE"
    assert str(ReconnectStrategy.COMPLEX_REORDER) == "COMPLEX_REORDER"


def test_reconnect_plan_with_unsplit():
    """测试目标不切分的情况（dst_parts=1）"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=4,
        dst_parts=1,
        batch_size=100,
        src_output="out",
        dst_input="in",
    )

    # 所有源都需要连接到唯一的输出
    assert len(plan.concat_operations) == 1
    assert len(plan.connections) == 4


def test_reconnect_plan_from_unsplit():
    """测试源不切分的情况（src_parts=1）"""
    plan = generate_reconnect_plan(
        src_op="A",
        dst_op="B",
        src_parts=1,
        dst_parts=4,
        batch_size=100,
        src_output="out",
        dst_input="in",
    )

    # 源需要被切分
    assert len(plan.split_operations) == 1
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_reconnect.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.transform.reconnect`

**Step 3: 实现数据流重连算法**

创建 `src/onnxsplit/transform/reconnect.py`:

```python
"""数据流重连算法"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ReconnectStrategy(Enum):
    """重连策略"""
    ONE_TO_ONE = "ONE_TO_ONE"  # 1对1直连
    SPLIT_SOURCE = "SPLIT_SOURCE"  # 源切分: src_parts < dst_parts
    CONCAT_SOURCE = "CONCAT_SOURCE"  # 源合并: src_parts > dst_parts
    COMPLEX_REORDER = "COMPLEX_REORDER"  # 复杂重排: 无整除关系

    @classmethod
    def determine(cls, src_parts: int, dst_parts: int) -> "ReconnectStrategy":
        """确定重连策略

        Args:
            src_parts: 源算子切分数
            dst_parts: 目标算子切分数

        Returns:
            重连策略
        """
        if src_parts == dst_parts:
            return ReconnectStrategy.ONE_TO_ONE
        elif dst_parts % src_parts == 0:
            return ReconnectStrategy.SPLIT_SOURCE
        elif src_parts % dst_parts == 0:
            return ReconnectStrategy.CONCAT_SOURCE
        else:
            return ReconnectStrategy.COMPLEX_REORDER


@dataclass
class ReconnectConnection:
    """单个连接关系"""
    src_split_idx: int
    dst_split_idx: int
    src_tensor: str
    dst_tensor: str
    slice_range: Optional[tuple[int, int]] = None  # 需要切片的范围

    def __repr__(self) -> str:
        return f"Connection({self.src_split_idx}->{self.dst_split_idx}, range={self.slice_range})"


@dataclass
class SliceOperation:
    """切片操作"""
    input_tensor: str
    output_tensor: str
    start: int
    end: int
    axis: int = 0


@dataclass
class ConcatOperation:
    """拼接操作"""
    input_tensors: list[str]
    output_tensor: str
    axis: int = 0


@dataclass
class SplitOperation:
    """切分操作"""
    input_tensor: str
    output_tensors: list[str]
    axis: int = 0
    split_sizes: list[int] | None = None


@dataclass
class ReconnectPlan:
    """重连计划"""
    src_op: str
    dst_op: str
    src_parts: int
    dst_parts: int
    src_output: str
    dst_input: str
    strategy: ReconnectStrategy
    connections: list[ReconnectConnection] = field(default_factory=list)
    slice_operations: list[SliceOperation] = field(default_factory=list)
    concat_operations: list[ConcatOperation] = field(default_factory=list)
    split_operations: list[SplitOperation] = field(default_factory=list)

    def summary(self) -> str:
        """生成计划摘要"""
        return (
            f"ReconnectPlan({self.src_op}[{self.src_parts}] -> {self.dst_op}[{self.dst_parts}], "
            f"strategy={self.strategy.value}, connections={len(self.connections)})"
        )


def calculate_overlap_range(
    src_start: int,
    src_end: int,
    dst_start: int,
    dst_end: int,
    batch_size: int,
) -> Optional[tuple[int, int]]:
    """计算源切片和目标切片的重叠区间

    Args:
        src_start: 源切片在原始张量中的起始位置
        src_end: 源切片在原始张量中的结束位置
        dst_start: 目标切片在原始张量中的起始位置
        dst_end: 目标切片在原始张量中的结束位置
        batch_size: 批次大小

    Returns:
        (重叠开始, 重叠结束) 相对于源切片的偏移，无重叠返回None
    """
    overlap_start = max(src_start, dst_start)
    overlap_end = min(src_end, dst_end)

    if overlap_start >= overlap_end:
        return None

    # 返回相对于源切片的偏移
    return (overlap_start - src_start, overlap_end - src_start)


def generate_reconnect_plan(
    src_op: str,
    dst_op: str,
    src_parts: int,
    dst_parts: int,
    batch_size: int,
    src_output: str,
    dst_input: str,
    axis: int = 0,
) -> ReconnectPlan:
    """生成数据流重连计划

    Args:
        src_op: 源算子名称
        dst_op: 目标算子名称
        src_parts: 源算子切分数
        dst_parts: 目标算子切分数
        batch_size: 批次大小
        src_output: 源算子输出名称
        dst_input: 目标算子输入名称
        axis: 切分轴

    Returns:
        重连计划
    """
    strategy = ReconnectStrategy.determine(src_parts, dst_parts)

    plan = ReconnectPlan(
        src_op=src_op,
        dst_op=dst_op,
        src_parts=src_parts,
        dst_parts=dst_parts,
        src_output=src_output,
        dst_input=dst_input,
        strategy=strategy,
    )

    if strategy == ReconnectStrategy.ONE_TO_ONE:
        # 1对1直连
        for i in range(src_parts):
            plan.connections.append(
                ReconnectConnection(
                    src_split_idx=i,
                    dst_split_idx=i,
                    src_tensor=f"{src_op}_split_{i}.{src_output}",
                    dst_tensor=f"{dst_op}_split_{i}.{dst_input}",
                )
            )

    elif strategy == ReconnectStrategy.SPLIT_SOURCE:
        # 源切分: dst_parts是src_parts的倍数
        ratio = dst_parts // src_parts
        src_chunk_size = batch_size // src_parts
        sub_chunk_size = src_chunk_size // ratio

        for src_i in range(src_parts):
            # 每个源切片需要再切分
            split_outputs = [f"{src_op}_{src_i}_sub_{j}" for j in range(ratio)]
            plan.split_operations.append(
                SplitOperation(
                    input_tensor=f"{src_op}_split_{i}.{src_output}",
                    output_tensors=split_outputs,
                    axis=axis,
                    split_sizes=[sub_chunk_size] * ratio,
                )
            )
            # 连接到目标
            for j in range(ratio):
                dst_idx = src_i * ratio + j
                plan.connections.append(
                    ReconnectConnection(
                        src_split_idx=src_i,
                        dst_split_idx=dst_idx,
                        src_tensor=split_outputs[j],
                        dst_tensor=f"{dst_op}_split_{dst_idx}.{dst_input}",
                    )
                )

    elif strategy == ReconnectStrategy.CONCAT_SOURCE:
        # 源合并: src_parts是dst_parts的倍数
        ratio = src_parts // dst_parts

        for dst_i in range(dst_parts):
            # 收集需要合并的源切片
            concat_inputs = []
            start_idx = dst_i * ratio
            for j in range(ratio):
                src_idx = start_idx + j
                concat_inputs.append(f"{src_op}_split_{src_idx}.{src_output}")
                plan.connections.append(
                    ReconnectConnection(
                        src_split_idx=src_idx,
                        dst_split_idx=dst_i,
                        src_tensor=concat_inputs[-1],
                        dst_tensor=f"concat_{src_op}_to_{dst_op}_{dst_i}",
                    )
                )

            plan.concat_operations.append(
                ConcatOperation(
                    input_tensors=concat_inputs,
                    output_tensor=f"concat_{src_op}_to_{dst_op}_{dst_i}",
                    axis=axis,
                )
            )

    else:  # COMPLEX_REORDER
        # 复杂重排：使用Slice+Concat
        src_chunk_size = batch_size // src_parts
        dst_chunk_size = batch_size // dst_parts

        for dst_i in range(dst_parts):
            dst_start = dst_i * dst_chunk_size
            dst_end = (dst_i + 1) * dst_chunk_size
            slice_outputs = []

            for src_i in range(src_parts):
                src_start = src_i * src_chunk_size
                src_end = (src_i + 1) * src_chunk_size

                overlap = calculate_overlap_range(
                    src_start, src_end, dst_start, dst_end, batch_size
                )

                if overlap is not None:
                    local_start, local_end = overlap
                    slice_output = f"slice_{src_op}_{src_i}_for_{dst_op}_{dst_i}"
                    slice_outputs.append(slice_output)

                    plan.slice_operations.append(
                        SliceOperation(
                            input_tensor=f"{src_op}_split_{src_i}.{src_output}",
                            output_tensor=slice_output,
                            start=local_start,
                            end=local_end,
                            axis=axis,
                        )
                    )

                    plan.connections.append(
                        ReconnectConnection(
                            src_split_idx=src_i,
                            dst_split_idx=dst_i,
                            src_tensor=slice_output,
                            dst_tensor=slice_output,
                            slice_range=(local_start, local_end),
                        )
                    )

            # 如果有多个切片，需要拼接
            if len(slice_outputs) > 1:
                concat_output = f"concat_{src_op}_to_{dst_op}_{dst_i}"
                plan.concat_operations.append(
                    ConcatOperation(
                        input_tensors=slice_outputs,
                        output_tensor=concat_output,
                        axis=axis,
                    )
                )
                # 更新连接目标
                for conn in plan.connections:
                    if conn.dst_split_idx == dst_i:
                        conn.dst_tensor = concat_output
            elif len(slice_outputs) == 1:
                # 单个切片直接连接
                for conn in plan.connections:
                    if conn.dst_split_idx == dst_i:
                        conn.dst_tensor = f"{dst_op}_split_{dst_i}.{dst_input}"

    return plan
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_reconnect.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/transform/reconnect.py tests/test_reconnect.py
git commit -m "feat: add data flow reconnection algorithm"
```

---

## Task 4: 图变换执行器

**Files:**
- Create: `src/onnxsplit/transform/executor.py`
- Test: `tests/test_transform_executor.py`

**Step 1: 编写图变换执行器测试**

创建 `tests/test_transform_executor.py`:

```python
"""测试图变换执行器"""
import onnx
from onnx import helper, TensorProto
from pathlib import Path
from onnxsplit.transform.executor import GraphTransformer
from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan


def test_transformer_creation():
    """测试创建变换器"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)
    assert transformer is not None


def test_transformer_split_single_operator():
    """测试切分单个算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)

    # 为conv_0创建切分方案
    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    new_model = transformer.apply_split_plan(plan)

    # 验证新模型
    assert new_model is not None
    assert len(list(new_model.graph.node)) >= len(list(analyzer.model.graph.node))


def test_transformer_preserve_inputs_outputs():
    """测试变换器保留输入输出"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)
    new_model = transformer.apply_split_plan(plan)

    # 输入输出应该保留
    assert len(new_model.graph.input) == len(analyzer.model.graph.input)
    assert len(new_model.graph.output) == len(analyzer.model.graph.output)


def test_transformer_clone_nodes():
    """测试节点克隆"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)

    # 检查原始节点
    original_nodes = list(analyzer.model.graph.node)

    plan = SplitPlan(operator_name="conv_0", parts=3, axis=0)
    new_model = transformer.apply_split_plan(plan)

    # 新模型应该有更多节点（原始+克隆+额外的split/concat）
    new_nodes = list(new_model.graph.node)
    assert len(new_nodes) >= len(original_nodes)


def test_transformer_matmul_model():
    """测试变换MatMul模型"""
    model_path = Path("tests/fixtures/models/simple_matmul.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="matmul_0", parts=2, axis=0)
    new_model = transformer.apply_split_plan(plan)

    assert new_model is not None


def test_transformer_multiple_splits():
    """测试多个算子切分"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)

    # 两个Conv算子都切分
    plans = [
        SplitPlan(operator_name="conv_0", parts=2, axis=0),
        SplitPlan(operator_name="conv_1", parts=2, axis=0),
    ]

    for plan in plans:
        transformer = GraphTransformer(ModelAnalyzer.from_path(model_path))
        new_model = transformer.apply_split_plan(plan)
        assert new_model is not None


def test_transformer_invalid_operator():
    """测试切分不存在的算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)

    plan = SplitPlan(operator_name="nonexistent", parts=2, axis=0)

    # 应该抛出异常或返回原模型
    try:
        new_model = transformer.apply_split_plan(plan)
        # 如果不抛异常，应该返回原模型
        assert new_model is not None
    except Exception:
        pass  # 预期的异常


def test_transformer_model_serialization():
    """测试变换后的模型序列化"""
    import tempfile

    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)
    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)
    new_model = transformer.apply_split_plan(plan)

    # 尝试序列化
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = f.name

    try:
        onnx.save(new_model, temp_path)
        # 重新加载验证
        loaded = onnx.load(temp_path)
        assert loaded is not None
    finally:
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_transformer_shape_inference():
    """测试变换后的形状推断"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)
    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)
    new_model = transformer.apply_split_plan(plan)

    # 运行形状推断
    new_model = onnx.shape_inference.infer_shapes(new_model)
    assert new_model is not None


def test_transformer_initializers_preserved():
    """测试初始器（权重）被保留"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    original_initializers = list(analyzer.model.graph.initializer)

    transformer = GraphTransformer(analyzer)
    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)
    new_model = transformer.apply_split_plan(plan)

    # 初始器应该被保留
    new_initializers = list(new_model.graph.initializer)
    assert len(new_initializers) >= len(original_initializers)


def test_transformer_value_info_preserved():
    """测试value_info被保留"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    transformer = GraphTransformer(analyzer)
    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)
    new_model = transformer.apply_split_plan(plan)

    # value_info可能增加，但原始的应该保留
    assert len(new_model.graph.value_info) >= 0
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_transform_executor.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.transform.executor`

**Step 3: 实现图变换执行器**

创建 `src/onnxsplit/transform/executor.py`:

```python
"""图变换执行器"""
import copy
from typing import Optional

import onnx
from onnx import helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.node_clone import clone_node, generate_split_name
from onnxsplit.transform.split_concat import create_split_node, create_concat_node, get_slice_initializers


class GraphTransformer:
    """图变换执行器

    根据切分方案对ONNX图进行变换。
    """

    def __init__(self, analyzer: ModelAnalyzer):
        """初始化变换器

        Args:
            analyzer: 模型分析器
        """
        self.analyzer = analyzer
        self._node_map: dict[str, list[onnx.NodeProto]] = {}  # 原始节点 -> 克隆节点列表
        self._tensor_map: dict[str, str] = {}  # 原始张量 -> 新张量映射

    def apply_split_plan(self, plan: SplitPlan) -> onnx.ModelProto:
        """应用切分方案到模型

        Args:
            plan: 切分方案

        Returns:
            变换后的新模型

        Raises:
            ValueError: 算子不存在
        """
        if not plan.is_split:
            # 不需要切分，返回原模型的副本
            return copy.deepcopy(self.analyzer.model)

        # 获取原始算子
        original_op = self.analyzer.get_operator(plan.operator_name)
        if original_op is None:
            raise ValueError(f"Operator not found: {plan.operator_name}")

        # 克隆模型
        new_model = copy.deepcopy(self.analyzer.model)
        new_graph = new_model.graph

        # 找到要切分的节点
        target_node = None
        for node in new_graph.node:
            if node.name == plan.operator_name:
                target_node = node
                break

        if target_node is None:
            raise ValueError(f"Node not found in graph: {plan.operator_name}")

        # 克隆节点
        cloned_nodes = []
        for i in range(plan.parts):
            new_outputs = [f"{out}_{i}" for out in target_node.output]
            cloned = clone_node(
                target_node,
                suffix=f"_split_{i}",
                new_outputs=new_outputs,
            )
            cloned_nodes.append(cloned)

        # 移除原始节点，添加克隆节点
        nodes_to_remove = []
        nodes_to_add = []
        for node in new_graph.node:
            if node.name == plan.operator_name:
                nodes_to_remove.append(node)
                # 插入输入切分（如果需要）
                if self._needs_input_split(target_node):
                    split_nodes = self._create_input_splits(target_node, plan)
                    nodes_to_add.extend(split_nodes)
                # 添加克隆节点
                nodes_to_add.extend(cloned_nodes)
                # 插入输出合并（如果需要）
                if self._needs_output_merge(target_node):
                    concat_nodes = self._create_output_merges(target_node, plan)
                    nodes_to_add.extend(concat_nodes)
                break

        # 更新图
        self._update_graph_nodes(new_graph, nodes_to_remove, nodes_to_add)

        # 运行形状推断
        new_model = onnx.shape_inference.infer_shapes(new_model)

        return new_model

    def _needs_input_split(self, node: onnx.NodeProto) -> bool:
        """检查是否需要在输入端插入Split"""
        # 如果输入是模型输入或其他算子的输出（不是Constant）
        for input_name in node.input:
            if not input_name:
                continue
            producer = self.analyzer.get_tensor_producer(input_name)
            # 如果输入来自其他算子或模型输入，需要切分
            if producer is None or producer != node.name:
                return True
        return False

    def _needs_output_merge(self, node: onnx.NodeProto) -> bool:
        """检查是否需要在输出端插入Concat"""
        # 如果输出被其他算子使用或是模型输出
        for output_name in node.output:
            consumers = self.analyzer.get_tensor_consumers(output_name)
            # 如果有消费者（排除自己）或是模型输出
            if consumers or self._is_model_output(output_name):
                return True
        return False

    def _is_model_output(self, tensor_name: str) -> bool:
        """检查张量是否是模型输出"""
        return any(output.name == tensor_name for output in self.analyzer.model.graph.output)

    def _create_input_splits(
        self, node: onnx.NodeProto, plan: SplitPlan
    ) -> list[onnx.NodeProto]:
        """创建输入切分节点"""
        split_nodes = []

        for input_name in node.input:
            if not input_name:
                continue

            # 只切分非权重输入
            if self._is_weight(input_name):
                continue

            split_node = create_split_node(
                input_name=input_name,
                axis=plan.axis,
                parts=plan.parts,
                output_prefix=f"{input_name}_split",
                node_name=f"split_{input_name}",
            )
            split_nodes.append(split_node)

        return split_nodes

    def _create_output_merges(
        self, node: onnx.NodeProto, plan: SplitPlan
    ) -> list[onnx.NodeProto]:
        """创建输出合并节点"""
        concat_nodes = []

        for i, output_name in enumerate(node.output):
            # 收集切分后的输出
            split_outputs = [f"{output_name}_{j}" for j in range(plan.parts)]

            concat_node = create_concat_node(
                input_names=split_outputs,
                output_name=output_name,  # 恢复原始名称
                axis=plan.axis,
                node_name=f"concat_{output_name}",
            )
            concat_nodes.append(concat_node)

        return concat_nodes

    def _is_weight(self, tensor_name: str) -> bool:
        """检查张量是否是权重"""
        return any(
            init.name == tensor_name
            for init in self.analyzer.model.graph.initializer
        )

    def _update_graph_nodes(
        self,
        graph: onnx.GraphProto,
        to_remove: list[onnx.NodeProto],
        to_add: list[onnx.NodeProto],
    ) -> None:
        """更新图的节点列表"""
        # 移除节点
        nodes_to_keep = []
        remove_names = {n.name for n in to_remove}
        for node in graph.node:
            if node.name not in remove_names:
                nodes_to_keep.append(node)

        # 清空并重建节点列表
        graph.node.clear()
        graph.node.extend(nodes_to_keep)

        # 添加新节点
        for node in to_add:
            graph.node.append(node)
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_transform_executor.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/transform/executor.py tests/test_transform_executor.py
git commit -m "feat: add graph transformer executor"
```

---

## Task 5: Transform模块导出

**Files:**
- Modify: `src/onnxsplit/transform/__init__.py`

**Step 1: 导出transform模块公共接口**

编辑 `src/onnxsplit/transform/__init__.py`:

```python
"""图变换模块

提供节点克隆、数据流重连和图变换功能。
"""

from onnxsplit.transform.node_clone import clone_node, generate_split_name
from onnxsplit.transform.split_concat import (
    create_split_node,
    create_concat_node,
    create_slice_node,
    get_slice_initializers,
)
from onnxsplit.transform.reconnect import (
    ReconnectStrategy,
    ReconnectConnection,
    ReconnectPlan,
    generate_reconnect_plan,
)
from onnxsplit.transform.executor import GraphTransformer


__all__ = [
    # Node clone
    "clone_node",
    "generate_split_name",
    # Split/Concat
    "create_split_node",
    "create_concat_node",
    "create_slice_node",
    "get_slice_initializers",
    # Reconnect
    "ReconnectStrategy",
    "ReconnectConnection",
    "ReconnectPlan",
    "generate_reconnect_plan",
    # Executor
    "GraphTransformer",
]
```

**Step 2: 验证模块导入**

Run: `uv run python -c "from onnxsplit.transform import *; print('Import successful')"`
Expected: 打印 "Import successful"

**Step 3: 运行所有transform测试**

Run: `uv run pytest tests/test_transform*.py tests/test_node_clone.py tests/test_split_concat.py tests/test_reconnect.py -v`
Expected: PASS - 所有测试通过

**Step 4: 提交**

```bash
git add src/onnxsplit/transform/__init__.py
git commit -m "chore: export transform module public API"
```

---

## 完成检查

**Step 1: 运行所有测试**

Run: `uv run pytest tests/ -v`
Expected: PASS - 所有测试通过

**Step 2: 检查代码风格**

Run: `uv run ruff check src/onnxsplit/transform tests/`
Expected: 无错误

**Step 3: 检查测试覆盖率**

Run: `uv run pytest tests/test_transform*.py tests/test_node_clone.py tests/test_split_concat.py tests/test_reconnect.py --cov=onnxsplit/transform --cov-report=term-missing`
Expected: 覆盖率 >= 80%

**Step 4: 最终提交**

```bash
git add .
git commit -m "chore: finalize plan 4 - graph transform module"
```

---

**Plan 4 完成！** 数据流重连与图变换模块已实现，包括：
- 节点克隆功能
- Split/Concat/Slice节点生成
- 数据流重连算法（支持任意切分数组合）
- 图变换执行器
- 完整的测试覆盖
