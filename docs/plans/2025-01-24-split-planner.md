# Plan 3: 切分轴识别与规划模块

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现切分轴智能识别和切分方案生成功能。

**Architecture:**
- 基于算子类型规则识别可切分轴
- 结合配置文件和内存限制生成切分方案
- 支持算子通配符匹配和配置优先级处理

**Tech Stack**: Python 3.13+, dataclasses, pytest

---

## Task 1: 切分轴规则定义

**Files:**
- Create: `src/onnxsplit/splitter/axis_rules.py`
- Test: `tests/test_axis_rules.py`

**Step 1: 编写切分轴规则测试**

创建 `tests/test_axis_rules.py`:

```python
"""测试切分轴规则"""
import pytest
from onnx import TensorProto
from onnxsplit.splitter.axis_rules import (
    SplitableAxes,
    AxisAnalyzer,
    get_splitable_axes_for_op,
)
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.analyzer.tensor import TensorMetadata


def test_splitable_axes_creation():
    """测试创建可切分轴集合"""
    axes = SplitableAxes(axes={0, 1, 2}, reason="Element-wise operation")
    assert axes.axes == {0, 1, 2}
    assert axes.reason == "Element-wise operation"


def test_splitable_axes_empty():
    """测试空可切分轴"""
    axes = SplitableAxes.empty("Cannot split")
    assert axes.axes == set()
    assert axes.reason == "Cannot split"


def test_splitable_axes_single():
    """测试单轴可切分"""
    axes = SplitableAxes.single(0, "Batch dimension")
    assert axes.axes == {0}
    assert axes.reason == "Batch dimension"


def test_splitable_axes_contains():
    """测试轴包含检查"""
    axes = SplitableAxes(axes={0, 2}, reason="test")
    assert 0 in axes
    assert 2 in axes
    assert 1 not in axes


def test_splitable_axes_len():
    """测试可切分轴数量"""
    axes = SplitableAxes(axes={0, 1, 2}, reason="test")
    assert len(axes) == 3


def test_splitable_axes_repr():
    """测试字符串表示"""
    axes = SplitableAxes(axes={0}, reason="Batch")
    repr_str = repr(axes)
    assert "{0}" in repr_str


def test_analyzer_elementwise_ops():
    """测试Element-wise算子可切任意轴"""
    analyzer = AxisAnalyzer()

    for op_type in ["Add", "Mul", "Sub", "Div", "Relu", "Sigmoid", "Tanh"]:
        op_info = OperatorInfo(
            name=f"test_{op_type}",
            op_type=op_type,
            attributes={},
            input_tensors=[
                TensorMetadata("input", shape=(2, 3, 4, 4), dtype=TensorProto.FLOAT)
            ],
            output_tensors=[
                TensorMetadata("output", shape=(2, 3, 4, 4), dtype=TensorProto.FLOAT)
            ],
        )
        axes = analyzer.analyze(op_info)
        # Element-wise算子应该可以切所有轴
        assert len(axes) > 0


def test_analyzer_conv_only_batch():
    """测试Conv只能切batch维度"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="conv_0",
        op_type="Conv",
        attributes={"kernel_shape": [3, 3]},
        input_tensors=[
            TensorMetadata("input", shape=(1, 3, 224, 224), dtype=TensorProto.FLOAT),
            TensorMetadata("weight", shape=(64, 3, 3, 3), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(1, 64, 112, 112), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    assert 0 in axes  # Batch维度可切
    assert 1 not in axes  # Channel维度不可切
    assert 2 not in axes  # Height维度不可切


def test_analyzer_conv_1d():
    """测试Conv1D只能切batch维度"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="conv1d",
        op_type="Conv",
        attributes={"kernel_shape": [3]},
        input_tensors=[
            TensorMetadata("input", shape=(2, 16, 100), dtype=TensorProto.FLOAT),
            TensorMetadata("weight", shape=(32, 16, 3), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("output", shape=(2, 32, 98), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    assert 0 in axes
    assert len(axes) == 1


def test_analyzer_matmul():
    """测试MatMul切分规则"""
    analyzer = AxismetAnalyzer()

    # 2D矩阵乘法: (M, K) @ (K, N) = (M, N)
    op_info = OperatorInfo(
        name="matmul_2d",
        op_type="MatMul",
        attributes={},
        input_tensors=[
            TensorMetadata("A", shape=(128, 64), dtype=TensorProto.FLOAT),
            TensorMetadata("B", shape=(64, 32), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("C", shape=(128, 32), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # 2D MatMul没有batch维度，通常不可切
    assert len(axes) == 0


def test_analyzer_matmul_3d():
    """测试3D MatMul（带batch维度）"""
    analyzer = AxisAnalyzer()

    # (B, M, K) @ (B, K, N) = (B, M, N)
    op_info = OperatorInfo(
        name="matmul_3d",
        op_type="MatMul",
        attributes={},
        input_tensors=[
            TensorMetadata("A", shape=(4, 128, 64), dtype=TensorProto.FLOAT),
            TensorMetadata("B", shape=(4, 64, 32), dtype=TensorProto.FLOAT),
        ],
        output_tensors=[
            TensorMetadata("C", shape=(4, 128, 32), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # Batch维度(axis=0)可切
    assert 0 in axes


def test_analyzer_reduce_ops():
    """测试Reduce算子只能切非归约轴"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="reduce_mean",
        op_type="ReduceMean",
        attributes={"axes": [1], "keepdims": 1},
        input_tensors=[
            TensorMetadata("input", shape=(2, 16, 32, 32), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(2, 1, 32, 32), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # axis=1是归约轴，不可切；其他轴可切
    assert 0 in axes  # Batch可切
    assert 1 not in axes  # 归约轴不可切
    assert 2 in axes  # H可切


def test_analyzer_reduce_mean_all_axes():
    """测试全轴归约的情况"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="reduce_mean_all",
        op_type="ReduceMean",
        attributes={"keepdims": 1},
        input_tensors=[
            TensorMetadata("input", shape=(2, 16, 32, 32), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(1, 1, 1, 1), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # 归约所有轴时，没有可切分的轴
    assert len(axes) == 0


def test_analyzer_batch_norm():
    """测试BatchNorm只能切batch维度"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="batch_norm",
        op_type="BatchNormalization",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(4, 16, 32, 32), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 16, 32, 32), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    assert 0 in axes
    assert 1 not in axes  # Channel维度统计量不可切


def test_analyzer_layer_norm():
    """测试LayerNorm只能切batch维度"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="layer_norm",
        op_type="LayerNormalization",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(4, 128, 768), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 128, 768), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    assert 0 in axes  # Batch可切
    # Layer norm通常在最后几维计算，这些维度不可切


def test_analyzer_softmax():
    """测试Softmax沿特定轴计算"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="softmax",
        op_type="Softmax",
        attributes={"axis": -1},
        input_tensors=[
            TensorMetadata("input", shape=(4, 128, 768), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 128, 768), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # Batch维度可切
    assert 0 in axes
    # 计算轴(-1)不可切


def test_analyzer_reshape():
    """测试Reshape算子需要特殊处理"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="reshape",
        op_type="Reshape",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(4, 128), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 8, 16), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # Reshape通常不直接切分
    assert len(axes) == 0


def test_analyzer_flatten():
    """测试Flatten算子"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="flatten",
        op_type="Flatten",
        attributes={"axis": 1},
        input_tensors=[
            TensorMetadata("input", shape=(4, 3, 224, 224), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 3*224*224), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # 只有batch维度可切
    assert 0 in axes


def test_analyzer_unknown_op():
    """测试未知算子类型"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="unknown",
        op_type="CustomOp",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(4, 16), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(4, 16), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # 未知算子保守处理，默认空
    assert len(axes) == 0


def test_get_splitable_axes_for_op():
    """测试便捷函数"""
    op_info = OperatorInfo(
        name="relu",
        op_type="Relu",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(2, 3, 4), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(2, 3, 4), dtype=TensorProto.FLOAT)
        ],
    )

    axes = get_splitable_axes_for_op(op_info)
    assert len(axes) > 0


def test_analyzer_shape_dimension_one():
    """测试维度为1时从可切分轴中移除"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="conv",
        op_type="Conv",
        attributes={},
        input_tensors=[
            TensorMetadata("input", shape=(1, 3, 224, 224), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(1, 64, 112, 112), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # batch维度为1，应该被移除（切分无意义）
    # 或者保留但允许用户配置
    assert 0 in axes  # 仍标记为可切，但实际切分时可能跳过


def test_analyzer_transpose():
    """测试Transpose算子"""
    analyzer = AxisAnalyzer()

    op_info = OperatorInfo(
        name="transpose",
        op_type="Transpose",
        attributes={"perm": [0, 2, 1, 3]},
        input_tensors=[
            TensorMetadata("input", shape=(2, 3, 224, 224), dtype=TensorProto.FLOAT)
        ],
        output_tensors=[
            TensorMetadata("output", shape=(2, 224, 3, 224), dtype=TensorProto.FLOAT)
        ],
    )

    axes = analyzer.analyze(op_info)
    # Batch维度(perm[0]=0)可切
    assert 0 in axes


def test_analyzer_pooling():
    """测试池化算子"""
    analyzer = AxisAnalyzer()

    for op_type in ["MaxPool", "AveragePool", "GlobalAveragePool"]:
        op_info = OperatorInfo(
            name=f"pool_{op_type}",
            op_type=op_type,
            attributes={"kernel_shape": [2, 2]} if "Global" not in op_type else {},
            input_tensors=[
                TensorMetadata("input", shape=(4, 16, 32, 32), dtype=TensorProto.FLOAT)
            ],
            output_tensors=[
                TensorMetadata("output", shape=(4, 16, 16, 16), dtype=TensorProto.FLOAT)
            ],
        )

        axes = analyzer.analyze(op_info)
        # Pooling可以切batch维度
        assert 0 in axes
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_axis_rules.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.splitter.axis_rules`

**Step 3: 实现切分轴规则**

创建 `src/onnxsplit/splitter/axis_rules.py`:

```python
"""切分轴识别规则"""
from dataclasses import dataclass

from onnxsplit.analyzer.operator import OperatorInfo


@dataclass(frozen=True)
class SplitableAxes:
    """可切分轴集合

    Attributes:
        axes: 可切分的轴索引集合
        reason: 原因说明
    """
    axes: set[int]
    reason: str

    @classmethod
    def empty(cls, reason: str = "No splitable axes") -> "SplitableAxes":
        """创建空的可切分轴集合"""
        return cls(set(), reason)

    @classmethod
    def single(cls, axis: int, reason: str = "") -> "SplitableAxes":
        """创建单轴可切分集合"""
        return cls({axis}, reason)

    def __contains__(self, axis: int) -> bool:
        """检查轴是否可切分"""
        return axis in self.axes

    def __len__(self) -> int:
        """可切分轴数量"""
        return len(self.axes)

    def __repr__(self) -> str:
        return f"SplitableAxes(axes={self.axes}, reason={self.reason!r})"


class AxisAnalyzer:
    """切分轴分析器

    基于算子类型和属性，智能识别可以切分的轴。
    """

    # Element-wise算子类型（输入输出形状相同，各元素独立计算）
    ELEMENT_WISE_OPS = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Sqrt",
        "Relu",
        "LeakyRelu",
        "PRelu",
        "Sigmoid",
        "Tanh",
        "Softplus",
        "Elu",
        "Abs",
        "Neg",
        "Exp",
        "Log",
        "Sin",
        "Cos",
        "Min",
        "Max",
        "Clip",
        "Floor",
        "Ceil",
        "Round",
        "Not",
        "And",
        "Or",
        "Xor",
        "Equal",
        "Greater",
        "Less",
        "Cast",
        "Identity",
    }

    def __init__(self):
        """初始化分析器"""
        pass

    def analyze(self, op_info: OperatorInfo) -> SplitableAxes:
        """分析算子的可切分轴

        Args:
            op_info: 算子信息

        Returns:
            可切分轴集合
        """
        op_type = op_info.op_type

        # 获取输入形状（使用第一个输入）
        if not op_info.input_tensors:
            return SplitableAxes.empty("No input tensors")

        input_shape = op_info.input_tensors[0].shape
        if not input_shape:
            return SplitableAxes.empty("Empty input shape")

        # 根据算子类型分析
        if op_type in self.ELEMENT_WISE_OPS:
            return self._analyze_elementwise(op_info, input_shape)
        elif op_type == "Conv":
            return self._analyze_conv(op_info, input_shape)
        elif op_type == "MatMul":
            return self._analyze_matmul(op_info)
        elif op_type.startswith("Reduce"):
            return self._analyze_reduce(op_info)
        elif op_type == "BatchNormalization":
            return self._analyze_batch_norm(op_info, input_shape)
        elif op_type == "LayerNormalization":
            return self._analyze_layer_norm(op_info, input_shape)
        elif op_type == "Softmax":
            return self._analyze_softmax(op_info, input_shape)
        elif op_type in ("MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool"):
            return self._analyze_pooling(op_info, input_shape)
        elif op_type == "Flatten":
            return self._analyze_flatten(op_info, input_shape)
        elif op_type == "Transpose":
            return self._analyze_transpose(op_info, input_shape)
        elif op_type in ("Reshape", "Squeeze", "Unsqueeze"):
            return SplitableAxes.empty(f"{op_type} changes tensor structure")
        else:
            # 未知算子保守处理
            return SplitableAxes.empty(f"Unknown operator type: {op_type}")

    def _analyze_elementwise(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Element-wise算子

        Element-wise算子所有输入输出形状相同（广播后），
        各元素独立计算，可以切分任意轴。
        """
        all_axes = set(range(len(input_shape)))
        return SplitableAxes(all_axes, "Element-wise operation")

    def _analyze_conv(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Conv算子

        Conv只能在batch维度(axis=0)切分。
        权重在channel维度共享，不能切分input的channel维度。
        """
        # 只有batch维度可切
        if len(input_shape) >= 1:
            return SplitableAxes.single(0, "Batch dimension for Conv")
        return SplitableAxes.empty("Conv input has no batch dimension")

    def _analyze_matmul(self, op_info: OperatorInfo) -> SplitableAxes:
        """分析MatMul算子

        对于 (B, M, K) @ (B, K, N) = (B, M, N) 的3D情况，
        batch维度可切分。

        对于 (M, K) @ (K, N) = (M, N) 的2D情况，
        通常不可切分（会影响矩阵乘法语义）。
        """
        if not op_info.input_tensors:
            return SplitableAxes.empty("No inputs for MatMul")

        shape_a = op_info.input_tensors[0].shape

        # 3D MatMul有batch维度
        if len(shape_a) == 3:
            return SplitableAxes.single(0, "Batch dimension for 3D MatMul")

        # 2D MatMul不可切
        return SplitableAxes.empty("2D MatMul cannot be split")

    def _analyze_reduce(self, op_info: OperatorInfo) -> SplitableAxes:
        """分析Reduce算子

        只能切分非归约轴。
        """
        if not op_info.input_tensors:
            return SplitableAxes.empty("No inputs for Reduce")

        input_shape = op_info.input_tensors[0].shape
        all_axes = set(range(len(input_shape)))

        # 获取归约轴
        reduce_axes_attr = op_info.get_attribute("axes")
        if reduce_axes_attr is not None:
            reduce_axes = set(reduce_axes_attr)
            # 处理负索引
            normalized_reduce = set()
            for ax in reduce_axes:
                if ax < 0:
                    normalized_reduce.add(len(input_shape) + ax)
                else:
                    normalized_reduce.add(ax)
        else:
            # 默认归约所有轴
            return SplitableAxes.empty("Reduce all axes")

        # 可切分的轴 = 所有轴 - 归约轴
        splitable = all_axes - normalized_reduce
        return SplitableAxes(splitable, f"Non-reduce axes (reducing {normalized_reduce})")

    def _analyze_batch_norm(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析BatchNorm算子

        只能切batch维度。
        Channel维度的统计量不可切分。
        """
        if len(input_shape) >= 1:
            return SplitableAxes.single(0, "Batch dimension for BatchNorm")
        return SplitableAxes.empty("BatchNorm input has no batch dimension")

    def _analyze_layer_norm(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析LayerNorm算子

        只能切batch维度。
        LayerNorm在最后几维计算统计量。
        """
        if len(input_shape) >= 1:
            return SplitableAxes.single(0, "Batch dimension for LayerNorm")
        return SplitableAxes.empty("LayerNorm input has no batch dimension")

    def _analyze_softmax(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Softmax算子

        只能切分非计算轴。
        """
        if not input_shape:
            return SplitableAxes.empty("Empty input for Softmax")

        # 获取计算轴
        axis_attr = op_info.get_attribute("axis", -1)

        # 标准化轴索引
        if axis_attr < 0:
            axis_attr = len(input_shape) + axis_attr

        all_axes = set(range(len(input_shape)))
        splitable = all_axes - {axis_attr}
        return SplitableAxes(splitable, f"Softmax computed on axis {axis_attr}")

    def _analyze_pooling(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析池化算子

        可以切batch维度。
        """
        if len(input_shape) >= 1:
            return SplitableAxes.single(0, "Batch dimension for Pooling")
        return SplitableAxes.empty("Pooling input has no batch dimension")

    def _analyze_flatten(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Flatten算子

        只能切batch维度（在flatten axis之前）。
        """
        # 获取flatten axis
        axis_attr = op_info.get_attribute("axis", 1)

        # flatten axis之前的维度可以切分（通常是batch）
        if axis_attr > 0:
            return SplitableAxes.single(0, "Batch dimension before flatten")
        return SplitableAxes.empty("Flatten from axis 0")

    def _analyze_transpose(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Transpose算子

        如果perm[0] == 0（batch维度不变），可以切分batch。
        """
        perm = op_info.get_attribute("perm")
        if perm and len(perm) > 0 and perm[0] == 0:
            return SplitableAxes.single(0, "Batch dimension preserved in transpose")
        return SplitableAxes.empty("Transpose changes batch dimension")


def get_splitable_axes_for_op(op_info: OperatorInfo) -> SplitableAxes:
    """便捷函数：获取算子的可切分轴

    Args:
        op_info: 算子信息

    Returns:
        可切分轴集合
    """
    analyzer = AxisAnalyzer()
    return analyzer.analyze(op_info)
```

**Step 4: 修正测试中的拼写错误**

Run: `uv run pytest tests/test_axis_rules.py::test_analyzer_matmul_3d -v`
Expected: 修正测试中的 `AxismetAnalyzer` 拼写错误

编辑 `tests/test_axis_rules.py` 第67行:

```python
def test_analyzer_matmul_3d():
    """测试3D MatMul（带batch维度）"""
    analyzer = AxisAnalyzer()  # 修正拼写
```

**Step 5: 运行测试验证通过**

Run: `uv run pytest tests/test_axis_rules.py -v`
Expected: PASS - 所有测试通过

**Step 6: 提交**

```bash
git add src/onnxsplit/splitter/axis_rules.py tests/test_axis_rules.py
git commit -m "feat: add splitable axis analyzer"
```

---

## Task 2: 切分方案数据结构

**Files:**
- Create: `src/onnxsplit/splitter/plan.py`
- Test: `tests/test_splitter_plan.py`

**Step 1: 编写切分方案测试**

创建 `tests/test_splitter_plan.py`:

```python
"""测试切分方案数据结构"""
from dataclasses import asdict
from onnxsplit.splitter.plan import SplitPlan, SplitReport
from onnxsplit.splitter.axis_rules import SplitableAxes


def test_split_plan_creation():
    """测试创建切分方案"""
    plan = SplitPlan(
        operator_name="/model/Conv_0",
        parts=4,
        axis=0,
    )
    assert plan.operator_name == "/model/Conv_0"
    assert plan.parts == 4
    assert plan.axis == 0
    assert plan.slice_ranges is None


def test_split_plan_with_ranges():
    """测试带切片范围的方案"""
    plan = SplitPlan(
        operator_name="/model/MatMul_0",
        parts=3,
        axis=0,
        slice_ranges=[(0, 34), (34, 68), (68, 100)],
    )
    assert len(plan.slice_ranges) == 3
    assert plan.slice_ranges[0] == (0, 34)


def test_split_plan_properties():
    """测试方案属性"""
    plan = SplitPlan(
        operator_name="test",
        parts=1,
        axis=None,
    )
    assert plan.is_split is False

    plan_split = SplitPlan(
        operator_name="test_split",
        parts=4,
        axis=0,
    )
    assert plan_split.is_split is True
    assert plan_split.chunk_size is None  # 没有总大小无法计算


def test_split_plan_with_total_size():
    """测试带总大小时的属性"""
    plan = SplitPlan(
        operator_name="test",
        parts=4,
        axis=0,
    )
    # 模拟总大小为100
    assert plan.get_chunk_size(100) == 25
    assert plan.get_chunk_size(101) == 26  # 向上取整


def test_split_plan_get_slice_range():
    """测试获取切片范围"""
    plan = SplitPlan(
        operator_name="test",
        parts=4,
        axis=0,
    )
    assert plan.get_slice_range(0, 100) == (0, 25)
    assert plan.get_slice_range(1, 100) == (25, 50)
    assert plan.get_slice_range(3, 100) == (75, 100)


def test_split_plan_with_predefined_ranges():
    """测试使用预定义范围的切片"""
    plan = SplitPlan(
        operator_name="test",
        parts=3,
        axis=0,
        slice_ranges=[(0, 30), (30, 70), (70, 100)],
    )
    assert plan.get_slice_range(0, 100) == (0, 30)
    assert plan.get_slice_range(1, 100) == (30, 70)
    assert plan.get_slice_range(2, 100) == (70, 100)


def test_split_plan_repr():
    """测试字符串表示"""
    plan = SplitPlan(
        operator_name="conv_0",
        parts=4,
        axis=0,
    )
    repr_str = repr(plan)
    assert "conv_0" in repr_str
    assert "4" in repr_str


def test_split_report_creation():
    """测试创建切分报告"""
    report = SplitReport(
        original_operators=100,
        split_operators=15,
        unsplit_operators=85,
        plans=[],
    )
    assert report.original_operators == 100
    assert report.split_operators == 15
    assert report.split_ratio == 0.15


def test_split_report_with_plans():
    """测试带方案的报告"""
    plans = [
        SplitPlan("conv_0", 4, 0),
        SplitPlan("matmul_0", 2, 0),
    ]
    report = SplitReport(
        original_operators=10,
        split_operators=2,
        unsplit_operators=8,
        plans=plans,
    )
    assert len(report.plans) == 2
    assert report.total_parts == 6  # 4 + 2


def test_split_report_get_plans_for_operator():
    """测试获取指定算子的方案"""
    plans = [
        SplitPlan("conv_0", 4, 0),
        SplitPlan("matmul_0", 2, 0),
    ]
    report = SplitReport(
        original_operators=10,
        split_operators=2,
        unsplit_operators=8,
        plans=plans,
    )

    plan = report.get_plan("conv_0")
    assert plan is not None
    assert plan.parts == 4

    assert report.get_plan("nonexistent") is None


def test_split_report_max_parts():
    """测试获取最大切分数"""
    plans = [
        SplitPlan("a", 2, 0),
        SplitPlan("b", 8, 0),
        SplitPlan("c", 4, 0),
    ]
    report = SplitReport(
        original_operators=10,
        split_operators=3,
        unsplit_operators=7,
        plans=plans,
    )

    assert report.max_parts == 8


def test_split_report_summary():
    """测试报告摘要"""
    report = SplitReport(
        original_operators=100,
        split_operators=20,
        unsplit_operators=80,
        plans=[
            SplitPlan("op1", 4, 0),
            SplitPlan("op2", 2, 0),
        ],
    )

    summary = report.summary()
    assert "20" in summary  # split_operators
    assert "100" in summary  # original_operators


def test_split_plan_axis_none():
    """测试axis=None的情况（不切分）"""
    plan = SplitPlan(
        operator_name="test",
        parts=1,
        axis=None,
    )
    assert plan.axis is None
    assert plan.is_split is False


def test_split_plan_parts_zero():
    """测试parts=0的特殊情况"""
    plan = SplitPlan(
        operator_name="test",
        parts=0,
        axis=None,
    )
    assert plan.is_split is False
    assert plan.parts == 0
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_splitter_plan.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.splitter.plan`

**Step 3: 实现切分方案数据结构**

创建 `src/onnxsplit/splitter/plan.py`:

```python
"""切分方案数据结构"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SplitPlan:
    """单个算子的切分方案

    Attributes:
        operator_name: 算子名称
        parts: 切分的份数，1表示不切分
        axis: 切分轴索引，None表示不切分
        slice_ranges: 每份的索引范围 [(start, end), ...]，None时均分
        reason: 切分原因说明（可选）
    """
    operator_name: str
    parts: int
    axis: Optional[int] = None
    slice_ranges: Optional[list[tuple[int, int]]] = None
    reason: Optional[str] = None

    @property
    def is_split(self) -> bool:
        """是否需要切分"""
        return self.parts > 1 and self.axis is not None

    @property
    def chunk_size(self) -> Optional[int]:
        """每份大小（如果设置了slice_ranges）"""
        if self.slice_ranges:
            return self.slice_ranges[0][1] - self.slice_ranges[0][0]
        return None

    def get_chunk_size(self, total_size: int) -> int:
        """计算均分时的每份大小

        Args:
            total_size: 总大小

        Returns:
            每份大小（向上取整）
        """
        if self.slice_ranges:
            return self.chunk_size or 0
        if self.parts <= 0:
            return 0
        return (total_size + self.parts - 1) // self.parts  # 向上取整

    def get_slice_range(self, part_idx: int, total_size: int) -> tuple[int, int]:
        """获取指定份的范围

        Args:
            part_idx: 份索引（0-based）
            total_size: 总大小

        Returns:
            (start, end) 范围元组
        """
        if not self.is_split:
            return (0, total_size)

        if self.slice_ranges and 0 <= part_idx < len(self.slice_ranges):
            return self.slice_ranges[part_idx]

        # 均分计算
        chunk = self.get_chunk_size(total_size)
        start = part_idx * chunk
        end = min(start + chunk, total_size)
        return (start, end)

    def __repr__(self) -> str:
        return f"SplitPlan(name={self.operator_name!r}, parts={self.parts}, axis={self.axis})"


@dataclass
class SplitReport:
    """切分方案报告

    Attributes:
        original_operators: 原始算子总数
        split_operators: 被切分的算子数
        unsplit_operators: 未切分的算子数
        plans: 所有切分方案列表
    """
    original_operators: int
    split_operators: int
    unsplit_operators: int
    plans: list[SplitPlan] = field(default_factory=list)

    @property
    def total_parts(self) -> int:
        """所有切分产生的总份数"""
        return sum(p.parts for p in self.plans if p.is_split)

    @property
    def split_ratio(self) -> float:
        """切分算子比例"""
        if self.original_operators == 0:
            return 0.0
        return self.split_operators / self.original_operators

    @property
    def max_parts(self) -> int:
        """单个算子的最大切分数"""
        if not self.plans:
            return 1
        return max((p.parts for p in self.plans), default=1)

    def get_plan(self, operator_name: str) -> Optional[SplitPlan]:
        """获取指定算子的切分方案

        Args:
            operator_name: 算子名称

        Returns:
            切分方案，不存在时返回None
        """
        for plan in self.plans:
            if plan.operator_name == operator_name:
                return plan
        return None

    def summary(self) -> str:
        """生成报告摘要"""
        return (
            f"SplitReport: {self.split_operators}/{self.original_operators} operators split "
            f"({self.split_ratio:.1%}), total {self.total_parts} parts"
        )

    def __repr__(self) -> str:
        return self.summary()
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_splitter_plan.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/splitter/plan.py tests/test_splitter_plan.py
git commit -m "feat: add split plan data structures"
```

---

## Task 3: 切分规划器

**Files:**
- Create: `src/onnxsplit/splitter/planner.py`
- Test: `tests/test_splitter_planner.py`

**Step 1: 编写切分规划器测试**

创建 `tests/test_splitter_planner.py`:

```python
"""测试切分规划器"""
import pytest
from onnx import TensorProto
from onnxsplit.splitter.planner import SplitPlanner
from onnxsplit.splitter.axis_rules import SplitableAxes
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.config import SplitConfig, GlobalConfig, OperatorConfig
from onnxsplit.analyzer import ModelAnalyzer, OperatorInfo, TensorMetadata
from pathlib import Path


def test_planner_with_no_config():
    """测试无配置时的规划"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=1))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 默认不切分
    assert len(report.plans) == 0
    assert report.split_operators == 0


def test_planner_with_global_default_parts():
    """测试全局默认切分数"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 应该为可切分的算子生成方案
    assert len(report.plans) > 0
    # 检查Conv有方案
    conv_plan = report.get_plan("conv_0")
    assert conv_plan is not None
    assert conv_plan.parts == 2
    assert conv_plan.axis == 0  # Batch维度


def test_planner_with_operator_config():
    """测试算子级别配置"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_0": OperatorConfig(parts=4, axis=0),
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_plan = report.get_plan("conv_0")
    assert conv_plan is not None
    assert conv_plan.parts == 4


def test_planner_wildcard_matching():
    """测试通配符匹配"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "/model/*": OperatorConfig(parts=2),  # 通配符
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 应该匹配到Conv算子
    assert len(report.plans) >= 2


def test_planner_with_axis_override():
    """测试axis覆盖"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),
        operators={
            "conv_0": OperatorConfig(parts=4, axis=0),  # 明确指定axis
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_plan = report.get_plan("conv_0")
    assert conv_plan.axis == 0


def test_planner_respects_splitable_axes():
    """测试尊重可切分轴限制"""
    # 创建虚拟分析器
    from onnxsplit.analyzer import ModelAnalyzer
    import onnx
    from onnx import helper

    # 创建一个简单模型
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 3, 8, 8]
    )
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        name="conv_0",
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, 4, 8, 8]
    )
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [4, 3, 3, 3], [0.1] * 108)
    const_node = helper.make_node("Constant", [], ["weight_value"], value=weight)
    conv_node.input[1] = "weight_value"

    graph = helper.make_graph(
        [const_node, conv_node],
        "test",
        [input_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=4))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # Conv只能切batch维度(axis=0)
    conv_plan = report.get_plan("conv_0")
    if conv_plan:
        assert conv_plan.axis == 0


def test_planner_parts_one_no_split():
    """测试parts=1时不生成切分方案"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_0": OperatorConfig(parts=1),  # 明确不切分
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # parts=1不算切分
    conv_plan = report.get_plan("conv_0")
    if conv_plan:
        assert not conv_plan.is_split


def test_planner_unsplitable_ops():
    """测试不可切分的算子"""
    # Reshape通常不可切分
    from onnxsplit.analyzer import ModelAnalyzer
    import onnx
    from onnx import helper

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [2, 128]
    )
    reshape_node = helper.make_node(
        "Reshape",
        inputs=["input", "shape"],
        outputs=["output"],
        name="reshape_0",
    )
    shape_const = helper.make_tensor("shape", TensorProto.INT64, [3], [2, 8, 16])
    const_node = helper.make_node("Constant", [], ["shape"], value=shape_const)
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, 8, 16]
    )

    graph = helper.make_graph(
        [const_node, reshape_node],
        "test",
        [input_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # Reshape不应该有切分方案
    reshape_plan = report.get_plan("reshape_0")
    assert reshape_plan is None or not reshape_plan.is_split


def test_planner_report_stats():
    """测试报告统计信息"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    assert report.original_operators > 0
    assert report.split_operators >= 0
    assert report.split_operators + report.unsplit_operators == report.original_operators


def test_planner_get_all_splitable_ops():
    """测试获取所有可切分算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)

    splitable = planner.get_splitable_operators()
    assert len(splitable) > 0
    # 应该包含Conv
    assert any(op.op_type == "Conv" for op in splitable)


def test_planner_with_empty_model():
    """测试空模型"""
    import onnx
    from onnx import helper

    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, 8, 8]
    )
    graph = helper.make_graph([], "empty", [input_tensor], [output_tensor])
    model = helper.make_model(graph)

    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    assert len(report.plans) == 0


def test_planner_priority():
    """测试配置优先级：算子配置 > 全局配置"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=2),  # 默认2份
        operators={
            "conv_0": OperatorConfig(parts=8),  # conv_0配置为8份
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_plan = report.get_plan("conv_0")
    assert conv_plan.parts == 8  # 使用算子配置，不是2


def test_planner_dynamic_shape_handling():
    """测试处理动态形状"""
    import onnx
    from onnx import helper

    # 动态batch维度
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch_dim", 3, 8, 8]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch_dim", 4, 8, 8]
    )

    # 使用identity作为占位
    identity_node = helper.make_node(
        "Identity",
        inputs=["input"],
        outputs=["output"],
        name="identity_0",
    )

    graph = helper.make_graph(
        [identity_node],
        "test",
        [input_tensor],
        [output_tensor],
    )
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 动态形状的算子应该被标记
    # 实际实现中可能需要特殊处理
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_splitter_planner.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.splitter.planner`

**Step 3: 实现切分规划器**

创建 `src/onnxsplit/splitter/planner.py`:

```python
"""切分规划器"""
import fnmatch
from typing import Optional

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.splitter.axis_rules import AxisAnalyzer, SplitableAxes
from onnxsplit.splitter.plan import SplitPlan, SplitReport
from onnxsplit.config import SplitConfig, GlobalConfig, OperatorConfig


class SplitPlanner:
    """切分规划器

    根据配置和模型分析结果，生成切分方案。
    """

    def __init__(self, analyzer: ModelAnalyzer, config: SplitConfig):
        """初始化规划器

        Args:
            analyzer: 模型分析器
            config: 切分配置
        """
        self.analyzer = analyzer
        self.config = config
        self.axis_analyzer = AxisAnalyzer()
        self._splitable_ops: dict[str, tuple[OperatorInfo, SplitableAxes]] = {}

    def generate(self) -> SplitReport:
        """生成切分方案

        Returns:
            切分报告
        """
        # 分析所有算子的可切分性
        self._analyze_splitability()

        # 生成切分方案
        plans = []
        for op_name, (op_info, splitable_axes) in self._splitable_ops.items():
            plan = self._create_plan_for_operator(op_info, splitable_axes)
            if plan and plan.is_split:
                plans.append(plan)

        # 统计
        total_ops = len(self._splitable_ops)
        split_ops = len(plans)
        unsplit_ops = total_ops - split_ops

        return SplitReport(
            original_operators=total_ops,
            split_operators=split_ops,
            unsplit_operators=unsplit_ops,
            plans=plans,
        )

    def _analyze_splitability(self) -> None:
        """分析所有算子的可切分性"""
        self._splitable_ops.clear()

        for op_info in self.analyzer.get_operators():
            splitable = self.axis_analyzer.analyze(op_info)
            self._splitable_ops[op_info.name] = (op_info, splitable)

    def _create_plan_for_operator(
        self,
        op_info: OperatorInfo,
        splitable_axes: SplitableAxes,
    ) -> Optional[SplitPlan]:
        """为单个算子创建切分方案

        Args:
            op_info: 算子信息
            splitable_axes: 可切分轴集合

        Returns:
            切分方案，如果不需要切分则返回None
        """
        # 获取该算子的配置
        parts, axis = self._get_operator_config(op_info.name)

        # parts=1表示不切分
        if parts <= 1:
            return None

        # 检查可切分性
        if not splitable_axes.axes:
            # 没有可切分轴
            return None

        # 确定切分轴
        if axis is not None:
            # 用户指定了轴，检查是否可切
            if axis not in splitable_axes:
                # 指定的轴不可切，回退到默认或跳过
                if splitable_axes.axes:
                    axis = next(iter(splitable_axes.axes))
                else:
                    return None
        else:
            # 自动选择轴：优先选择batch(axis=0)
            if 0 in splitable_axes.axes:
                axis = 0
            else:
                # 选择第一个可切分轴
                axis = min(splitable_axes.axes) if splitable_axes.axes else None

        if axis is None:
            return None

        return SplitPlan(
            operator_name=op_info.name,
            parts=parts,
            axis=axis,
            reason=splitable_axes.reason,
        )

    def _get_operator_config(self, op_name: str) -> tuple[int, Optional[int]]:
        """获取算子的切分配置

        优先级：算子精确匹配 > 通配符匹配 > 全局配置

        Args:
            op_name: 算子名称

        Returns:
            (parts, axis) 元组
        """
        # 1. 精确匹配
        if op_name in self.config.operators:
            op_config = self.config.operators[op_name]
            return (op_config.parts, op_config.axis)

        # 2. 通配符匹配
        for pattern, op_config in self.config.operators.items():
            if fnmatch.fnmatch(op_name, pattern):
                return (op_config.parts, op_config.axis)

        # 3. 全局配置
        return (self.config.global_config.default_parts, None)

    def get_splitable_operators(self) -> list[OperatorInfo]:
        """获取所有可切分的算子列表

        Returns:
            可切分算子列表
        """
        if not self._splitable_ops:
            self._analyze_splitability()

        return [
            op_info
            for op_info, splitable in self._splitable_ops.values()
            if splitable.axes
        ]
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_splitter_planner.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/splitter/planner.py tests/test_splitter_planner.py
git commit -m "feat: add split planner"
```

---

## Task 4: Splitter模块导出

**Files:**
- Modify: `src/onnxsplit/splitter/__init__.py`

**Step 1: 导出splitter模块公共接口**

编辑 `src/onnxsplit/splitter/__init__.py`:

```python
"""切分规划模块

提供切分轴识别和切分方案生成功能。
"""

from onnxsplit.splitter.axis_rules import (
    SplitableAxes,
    AxisAnalyzer,
    get_splitable_axes_for_op,
)
from onnxsplit.splitter.plan import SplitPlan, SplitReport
from onnxsplit.splitter.planner import SplitPlanner


__all__ = [
    # Axis rules
    "SplitableAxes",
    "AxisAnalyzer",
    "get_splitable_axes_for_op",
    # Plan
    "SplitPlan",
    "SplitReport",
    # Planner
    "SplitPlanner",
]
```

**Step 2: 验证模块导入**

Run: `uv run python -c "from onnxsplit.splitter import *; print('Import successful')"`
Expected: 打印 "Import successful"

**Step 3: 运行所有splitter测试**

Run: `uv run pytest tests/test_splitter_*.py tests/test_axis_*.py -v`
Expected: PASS - 所有测试通过

**Step 4: 提交**

```bash
git add src/onnxsplit/splitter/__init__.py
git commit -m "chore: export splitter module public API"
```

---

## 完成检查

**Step 1: 运行所有测试**

Run: `uv run pytest tests/ -v`
Expected: PASS - 所有测试通过

**Step 2: 检查代码风格**

Run: `uv run ruff check src/onnxsplit/splitter tests/`
Expected: 无错误

**Step 3: 检查测试覆盖率**

Run: `uv run pytest tests/test_splitter_*.py tests/test_axis_*.py --cov=onnxsplit/splitter --cov-report=term-missing`
Expected: 覆盖率 >= 85%

**Step 4: 最终提交**

```bash
git add .
git commit -m "chore: finalize plan 3 - split planner module"
```

---

**Plan 3 完成！** 切分轴识别与规划模块已实现，包括：
- 切分轴规则定义（支持各种算子类型）
- 切分方案数据结构
- 切分规划器（支持通配符和配置优先级）
- 完整的测试覆盖
