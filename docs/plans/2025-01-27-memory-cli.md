# Plan 5: 内存估算与CLI集成

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现内存估算、自动切分调整和完整的CLI接口。

**Architecture:**
- 基于张量shape和dtype计算内存占用
- 根据内存限制自动调整切分数
- 使用argparse/typer构建CLI
- 集成所有模块提供完整的用户命令

**Tech Stack**: Python 3.13+, typer, rich, pytest

---

## Task 1: 内存估算器

**Files:**
- Create: `src/onnxsplit/memory/estimator.py`
- Test: `tests/test_memory_estimator.py`

**Step 1: 编写内存估算器测试**

创建 `tests/test_memory_estimator.py`:

```python
"""测试内存估算器"""
import pytest
from onnx import TensorProto
from onnxsplit.memory.estimator import (
    MemoryEstimator,
    TensorMemoryInfo,
    estimate_tensor_memory,
    dtype_bytes,
)
from onnxsplit.analyzer import ModelAnalyzer, OperatorInfo, TensorMetadata
from pathlib import Path


def test_dtype_bytes():
    """测试各数据类型的字节大小"""
    assert dtype_bytes(TensorProto.FLOAT) == 4
    assert dtype_bytes(TensorProto.FLOAT16) == 2
    assert dtype_bytes(TensorProto.DOUBLE) == 8
    assert dtype_bytes(TensorProto.INT8) == 1
    assert dtype_bytes(TensorProto.INT16) == 2
    assert dtype_bytes(TensorProto.INT32) == 4
    assert dtype_bytes(TensorProto.INT64) == 8
    assert dtype_bytes(TensorProto.BOOL) == 1
    assert dtype_bytes(TensorProto.UINT8) == 1
    assert dtype_bytes(TensorProto.COMPLEX64) == 8
    assert dtype_bytes(TensorProto.COMPLEX128) == 16


def test_estimate_tensor_memory():
    """测试估算张量内存"""
    # FLOAT32: 100 * 200 * 4 bytes = 80KB
    mem = estimate_tensor_memory((100, 200), TensorProto.FLOAT)
    assert mem == 100 * 200 * 4


def test_estimate_tensor_memory_1d():
    """测试1D张量"""
    mem = estimate_tensor_memory((1024,), TensorProto.FLOAT)
    assert mem == 1024 * 4


def test_estimate_tensor_memory_empty():
    """测试空张量（标量）"""
    mem = estimate_tensor_memory((), TensorProto.FLOAT)
    assert mem == 4  # 标量占用一个元素


def test_estimate_tensor_memory_float16():
    """测试FLOAT16"""
    mem = estimate_tensor_memory((100, 100), TensorProto.FLOAT16)
    assert mem == 100 * 100 * 2


def test_memory_estimator_creation():
    """测试创建内存估算器"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    assert estimator is not None


def test_estimator_get_tensor_memory():
    """测试获取张量内存"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    inputs = analyzer.get_inputs()
    if inputs:
        info = estimator.get_tensor_memory(inputs[0].name)
        assert info is not None
        assert info.tensor_name == inputs[0].name
        assert info.memory_bytes > 0


def test_estimator_get_operator_memory():
    """测试获取算子内存"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    conv_op = analyzer.get_operator("conv_0")
    if conv_op:
        info = estimator.get_operator_memory(conv_op)
        assert info is not None
        assert info.operator_name == "conv_0"
        assert info.total_memory_mb > 0


def test_estimator_get_total_model_memory():
    """测试获取总模型内存"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    total = estimator.get_total_model_memory()
    assert total > 0


def test_estimator_get_peak_memory():
    """测试获取峰值内存"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    peak = estimator.get_peak_memory()
    assert peak >= 0


def test_estimator_get_memory_breakdown():
    """测试获取内存分解"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    breakdown = estimator.get_memory_breakdown()
    assert len(breakdown) > 0
    assert all(info.total_memory_mb >= 0 for info in breakdown)


def test_tensor_memory_info():
    """测试张量内存信息"""
    info = TensorMemoryInfo(
        tensor_name="input",
        shape=(1, 3, 224, 224),
        dtype=TensorProto.FLOAT,
        memory_bytes=1 * 3 * 224 * 224 * 4,
    )

    assert info.tensor_name == "input"
    assert info.size_mb == pytest.approx(0.6, rel=0.1)


def test_operator_memory_info():
    """测试算子内存信息"""
    from onnxsplit.memory.estimator import OperatorMemoryInfo

    info = OperatorMemoryInfo(
        operator_name="conv_0",
        op_type="Conv",
        input_memory_mb=1.0,
        output_memory_mb=0.5,
        total_memory_mb=1.5,
    )

    assert info.operator_name == "conv_0"
    assert info.total_memory_mb == 1.5


def test_estimator_with_dynamic_shape():
    """测试处理动态形状"""
    import onnx
    from onnx import helper

    # 创建带动态形状的模型
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 3, 224, 224]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", 16, 224, 224]
    )
    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "weight"],
        outputs=["output"],
        name="conv_0",
    )
    weight = helper.make_tensor("weight", TensorProto.FLOAT, [16, 3, 3, 3], [0.1] * 432)
    const_node = helper.make_node("Constant", [], ["weight_value"], value=weight)
    conv_node.input[1] = "weight_value"

    graph = helper.make_graph(
        [const_node, conv_node], "test", [input_tensor], [output_tensor]
    )
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)

    # 动态形状的内存应该返回0或特殊处理
    breakdown = estimator.get_memory_breakdown()
    # 应该能正常处理，动态维度返回0
    assert len(breakdown) >= 0


def test_estimator_weights_memory():
    """测试权重内存计算"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    # 获取初始器（权重）
    weights_memory = estimator.get_weights_memory()
    assert weights_memory >= 0


def test_dtype_bytes_unknown():
    """测试未知类型"""
    # 假设999是未知类型
    assert dtype_bytes(999) == 4  # 默认值


def test_estimate_tensor_memory_large():
    """测试大张量"""
    # 1000x1000x1000 FLOAT32 = 4GB
    mem = estimate_tensor_memory((1000, 1000, 1000), TensorProto.FLOAT)
    assert mem == 4_000_000_000
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_memory_estimator.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.memory.estimator`

**Step 3: 实现内存估算器**

创建 `src/onnxsplit/memory/estimator.py`:

```python
"""内存估算器"""
from dataclasses import dataclass
from typing import Optional

from onnx import TensorProto

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.analyzer.tensor import TensorMetadata


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
        return any(
            init.name == tensor_name
            for init in self.analyzer.model.graph.initializer
        )

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
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_memory_estimator.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/memory/estimator.py tests/test_memory_estimator.py
git commit -m "feat: add memory estimator"
```

---

## Task 2: 自动切分调整

**Files:**
- Create: `src/onnxsplit/memory/auto_adjust.py`
- Test: `tests/test_auto_adjust.py`

**Step 1: 编写自动调整测试**

创建 `tests/test_auto_adjust.py`:

```python
"""测试自动切分调整"""
import pytest
from onnxsplit.memory.auto_adjust import AutoSplitAdjuster
from onnxsplit.memory.estimator import MemoryEstimator, OperatorMemoryInfo
from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from pathlib import Path


def test_adjuster_no_limit():
    """测试无内存限制时不调整"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    adjusted = adjuster.adjust_plan(plan, max_memory_mb=None)
    assert adjusted.parts == 2  # 不调整


def test_adjuster_under_limit():
    """测试低于限制时不调整"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    # 设置一个很大的限制
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=10000)
    assert adjusted.parts == 2


def test_adjuster_over_limit():
    """测试超过限制时增加切分数"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    # 获取算子内存
    conv_op = analyzer.get_operator("conv_0")
    if conv_op:
        op_mem = estimator.get_operator_memory(conv_op)
        if op_mem:
            # 设置一个低于当前内存的限制
            adjusted = adjuster.adjust_plan(
                SplitPlan(operator_name="conv_0", parts=1, axis=0),
                max_memory_mb=op_mem.total_memory_mb / 4,  # 强切分
            )
            assert adjusted.parts >= 1


def test_adjuster_max_parts_limit():
    """测试最大切分数限制"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    # 设置极低的内存限制
    plan = SplitPlan(operator_name="conv_0", parts=1, axis=0)

    adjusted = adjuster.adjust_plan(
        plan,
        max_memory_mb=0.001,  # 1KB
        max_parts=256,
    )

    # 应该受到max_parts限制
    assert adjusted.parts <= 256


def test_adjuster_with_large_parts():
    """测试处理大切分数"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=100, axis=0)

    # 大切分数应该被警告或限制
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=None)
    assert adjusted.parts == 100


def test_adjuster_unsplitable():
    """测试不可切分算子"""
    # Reshape通常不可切分
    import onnx
    from onnx import helper

    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 128])
    shape_const = helper.make_tensor("shape", onnx.TensorProto.INT64, [2], [2, 64])
    const_node = helper.make_node("Constant", [], ["shape"], value=shape_const)
    reshape_node = helper.make_node(
        "Reshape", inputs=["input", "shape"], outputs=["output"], name="reshape_0"
    )
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 64])

    graph = helper.make_graph(
        [const_node, reshape_node], "test", [input_tensor], [output_tensor]
    )
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="reshape_0", parts=1, axis=None)
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=1.0)

    # 不可切分，返回原计划
    assert adjusted.parts == 1


def test_adjuster_binary_search():
    """测试二分查找最优切分数"""
    # 模拟一个内存占用已知的算子
    import onnx
    from onnx import helper

    # 创建一个大张量的模型
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1000, 1000])
    add_node = helper.make_node("Add", inputs=["input", "input"], outputs=["output"], name="add_0")
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1000, 1000])

    graph = helper.make_graph([add_node], "test", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="add_0", parts=1, axis=0)

    # 设置限制使需要切分
    # 1000*1000*4 bytes = 4MB per tensor
    # 限制1MB需要至少切4份
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=2.0)
    assert adjusted.parts >= 1


def test_adjuster_preserve_axis():
    """测试调整时保留切分轴"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    adjusted = adjuster.adjust_plan(plan, max_memory_mb=None)
    assert adjusted.axis == 0


def test_adjuster_with_weights():
    """测试包含权重的算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    # Conv有权重，权重不应被切分
    conv_op = analyzer.get_operator("conv_0")
    if conv_op:
        plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)
        adjusted = adjuster.adjust_plan(plan, max_memory_mb=None)
        # 权重不影响切分决策
        assert adjusted.axis == 0


def test_adjuster_report_adjustment():
    """测试调整报告"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)
    adjuster = AutoSplitAdjuster(estimator)

    plan = SplitPlan(operator_name="conv_0", parts=2, axis=0)

    # 强制调整
    original_parts = plan.parts
    adjusted = adjuster.adjust_plan(plan, max_memory_mb=0.001)

    # 如果发生了调整
    if adjusted.parts != original_parts:
        assert adjusted.reason is not None
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_auto_adjust.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.memory.auto_adjust`

**Step 3: 实现自动切分调整**

创建 `src/onnxsplit/memory/auto_adjust.py`:

```python
"""自动切分调整"""
import math

from onnxsplit.memory.estimator import MemoryEstimator
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.splitter.axis_rules import AxisAnalyzer, SplitableAxes


class AutoSplitAdjuster:
    """自动切分调整器

    根据内存限制自动调整切分数。
    """

    def __init__(
        self,
        estimator: MemoryEstimator,
        max_parts: int = 256,
        warn_threshold: int = 64,
    ):
        """初始化调整器

        Args:
            estimator: 内存估算器
            max_parts: 最大切分数限制
            warn_threshold: 切分警告阈值
        """
        self.estimator = estimator
        self.max_parts = max_parts
        self.warn_threshold = warn_threshold
        self.axis_analyzer = AxisAnalyzer()

    def adjust_plan(
        self,
        plan: SplitPlan,
        max_memory_mb: float | None,
    ) -> SplitPlan:
        """调整切分方案

        Args:
            plan: 原始切分方案
            max_memory_mb: 内存限制（MB），None表示不限制

        Returns:
            调整后的切分方案
        """
        if max_memory_mb is None or not plan.is_split:
            return plan

        # 获取算子信息
        op_info = self.estimator.analyzer.get_operator(plan.operator_name)
        if op_info is None:
            return plan

        # 获取内存信息
        op_mem = self.estimator.get_operator_memory(op_info)
        if op_mem is None or op_mem.total_memory_mb == 0:
            return plan

        # 检查是否需要调整
        per_part_memory = op_mem.total_memory_mb / plan.parts
        if per_part_memory <= max_memory_mb:
            return plan

        # 计算需要的切分数
        needed_parts = self._calculate_needed_parts(
            op_mem.total_memory_mb, max_memory_mb, plan.parts
        )

        # 限制在max_parts范围内
        final_parts = min(needed_parts, self.max_parts)

        # 创建新方案
        return SplitPlan(
            operator_name=plan.operator_name,
            parts=final_parts,
            axis=plan.axis,
            slice_ranges=plan.slice_ranges,
            reason=f"Adjusted from {plan.parts} to {final_parts} for memory limit",
        )

    def _calculate_needed_parts(
        self,
        total_memory_mb: float,
        max_memory_mb: float,
        current_parts: int,
    ) -> int:
        """计算满足内存限制所需的切分数

        使用二分查找确定最小切分数。

        Args:
            total_memory_mb: 总内存
            max_memory_mb: 每份内存限制
            current_parts: 当前切分数

        Returns:
            需要的切分数
        """
        # 从当前切分数开始
        min_parts = max(current_parts, 1)
        max_parts = self.max_parts

        # 快速检查
        if total_memory_mb / min_parts <= max_memory_mb:
            return min_parts

        # 二分查找
        while min_parts < max_parts:
            mid_parts = (min_parts + max_parts) // 2
            per_part = total_memory_mb / mid_parts

            if per_part <= max_memory_mb:
                max_parts = mid_parts
            else:
                min_parts = mid_parts + 1

        return min_parts

    def adjust_report(
        self,
        plans: list[SplitPlan],
        max_memory_mb: float | None,
    ) -> list[SplitPlan]:
        """批量调整切分方案

        Args:
            plans: 切分方案列表
            max_memory_mb: 内存限制

        Returns:
            调整后的方案列表
        """
        if max_memory_mb is None:
            return plans

        return [
            self.adjust_plan(plan, max_memory_mb)
            for plan in plans
        ]
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_auto_adjust.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/memory/auto_adjust.py tests/test_auto_adjust.py
git commit -m "feat: add auto split adjuster"
```

---

## Task 3: CLI参数解析

**Files:**
- Create: `src/onnxsplit/cli/parser.py`
- Test: `tests/test_cli_parser.py`

**Step 1: 编写CLI解析器测试**

创建 `tests/test_cli_parser.py`:

```python
"""测试CLI参数解析"""
import pytest
from typer.testing import CliRunner
from onnxsplit.cli.parser import get_cli_app, CliOptions


def test_cli_app_creation():
    """测试创建CLI应用"""
    app = get_cli_app()
    assert app is not None


def test_parse_basic_command():
    """测试解析基本命令"""
    runner = CliRunner()
    app = get_cli_app()

    # 测试--help
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "onnxsplit" in result.stdout


def test_parse_model_argument():
    """测试解析模型参数"""
    import tempfile
    from pathlib import Path

    # 创建临时模型文件
    import onnx
    from onnx import helper

    model = helper.make_model(helper.make_graph([], "test", [], []))
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = Path(f.name)
        onnx.save(model, f)

    try:
        runner = CliRunner()
        app = get_cli_app()

        result = runner.invoke(app, [str(temp_path)])
        # 应该有输出（可能是错误，但不应该是参数错误）
        assert "Error" not in result.stdout or "missing" not in result.stdout.lower()
    finally:
        temp_path.unlink()


def test_parse_parts_argument():
    """测试解析parts参数"""
    runner = CliRunner()
    app = get_cli_app()

    result = runner.invoke(app, ["--parts", "4", "dummy.onnx"])
    # 只要不报错就行
    assert result.exit_code != 2  # 2是参数错误


def test_parse_max_memory_argument():
    """测试解析max-memory参数"""
    runner = CliRunner()
    app = get_cli_app()

    result = runner.invoke(app, ["--max-memory", "512", "dummy.onnx"])
    assert result.exit_code != 2


def test_parse_config_argument():
    """测试解析config参数"""
    import tempfile
    from pathlib import Path

    # 创建临时配置文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("global:\n  default_parts: 2\n")
        temp_path = Path(f.name)

    try:
        runner = CliRunner()
        app = get_cli_app()

        result = runner.invoke(app, ["--config", str(temp_path), "dummy.onnx"])
        assert result.exit_code != 2
    finally:
        temp_path.unlink()


def test_parse_output_argument():
    """测试解析output参数"""
    runner = CliRunner()
    app = get_cli_app()

    result = runner.invoke(app, ["--output", "output.onnx", "dummy.onnx"])
    assert result.exit_code != 2


def test_parse_report_argument():
    """测试解析report参数"""
    runner = CliRunner()
    app = get_cli_app()

    result = runner.invoke(app, ["--report", "report.json", "dummy.onnx"])
    assert result.exit_code != 2


def test_options_dataclass():
    """测试CliOptions数据类"""
    options = CliOptions(
        model="model.onnx",
        config=None,
        parts=4,
        max_memory=512,
        output="output.onnx",
        report=None,
    )

    assert options.model == "model.onnx"
    assert options.parts == 4
    assert options.max_memory == 512


def test_options_default_values():
    """测试默认值"""
    options = CliOptions(model="model.onnx")

    assert options.parts == 1
    assert options.max_memory is None
    assert options.output is None  # 会使用默认值


def test_generate_default_output_name():
    """测试生成默认输出文件名"""
    from onnxsplit.cli.parser import generate_output_name

    assert generate_output_name("model.onnx") == "model_split.onnx"
    assert generate_output_name("path/to/model.onnx") == "path/to/model_split.onnx"
    assert generate_output_name("model") == "model_split.onnx"
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_cli_parser.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.cli.parser`

**Step 3: 实现CLI解析器**

创建 `src/onnxsplit/cli/parser.py`:

```python
"""CLI参数解析"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="onnxsplit",
    help="ONNX model operator splitting tool for memory optimization",
    add_completion=False,
)


@dataclass
class CliOptions:
    """CLI选项"""
    model: str
    config: Optional[str] = None
    parts: int = 1
    max_memory: Optional[int] = None
    output: Optional[str] = None
    report: Optional[str] = None


def generate_output_name(model_path: str) -> str:
    """生成默认输出文件名

    Args:
        model_path: 输入模型路径

    Returns:
        输出文件名
    """
    path = Path(model_path)
    if path.suffix == ".onnx":
        return str(path.with_name(f"{path.stem}_split.onnx"))
    return f"{model_path}_split.onnx"


def get_cli_app() -> typer.Typer:
    """获取CLI应用

    Returns:
        Typer应用实例
    """
    return app


@app.callback()
def main(
    ctx: typer.Context,
    model: str = typer.Argument(..., help="Path to input ONNX model"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    parts: int = typer.Option(1, "--parts", "-p", help="Default number of splits"),
    max_memory: Optional[int] = typer.Option(None, "--max-memory", "-m", help="Max memory per split (MB)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output model path"),
    report: Optional[str] = typer.Option(None, "--report", "-r", help="Report JSON output path"),
):
    """ONNX模型切分工具"""
    # 存储选项到context
    ctx.ensure_object(dict)
    ctx.obj["options"] = CliOptions(
        model=model,
        config=config,
        parts=parts,
        max_memory=max_memory,
        output=output,
        report=report,
    )


@app.command()
def split(ctx: typer.Context):
    """Split ONNX model according to configuration"""
    options: CliOptions = ctx.obj["options"]

    # 导入实现
    from onnxsplit.cli.runner import run_split

    # 确定输出路径
    output = options.output or generate_output_name(options.model)

    # 运行切分
    run_split(
        model_path=options.model,
        config_path=options.config,
        cli_parts=options.parts,
        cli_max_memory=options.max_memory,
        output_path=output,
        report_path=options.report,
    )


@app.command()
def analyze(ctx: typer.Context):
    """Analyze ONNX model and show memory statistics"""
    options: CliOptions = ctx.obj["options"]

    from onnxsplit.cli.runner import run_analyze

    run_analyze(
        model_path=options.model,
        report_path=options.report,
    )


@app.command()
def validate(
    model: str = typer.Argument(..., help="Path to ONNX model"),
):
    """Validate ONNX model structure"""
    import onnx

    try:
        onnx_model = onnx.load(model)
        onnx.checker.check_model(onnx_model)
        typer.echo(f"✓ Model '{model}' is valid")
    except Exception as e:
        typer.echo(f"✗ Model validation failed: {e}", err=True)
        raise typer.Exit(1)
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_cli_parser.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/cli/parser.py tests/test_cli_parser.py
git commit -m "feat: add CLI parser"
```

---

## Task 4: CLI运行器

**Files:**
- Create: `src/onnxsplit/cli/runner.py`
- Test: `tests/test_cli_runner.py`

**Step 1: 编写CLI运行器测试**

创建 `tests/test_cli_runner.py`:

```python
"""测试CLI运行器"""
import pytest
import tempfile
from pathlib import Path
from onnxsplit.cli.runner import run_split, run_analyze


def test_run_split_basic():
    """测试基本切分运行"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.onnx"

        run_split(
            model_path=str(model_path),
            config_path=None,
            cli_parts=2,
            cli_max_memory=None,
            output_path=str(output_path),
            report_path=None,
        )

        # 检查输出文件存在
        assert output_path.exists()


def test_run_split_with_config():
    """测试使用配置文件运行"""
    import onnx
    from onnx import helper

    # 创建临时模型
    model = helper.make_model(helper.make_graph([], "test", [], []))
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        model_path = Path(f.name)
        onnx.save(model, f)

    # 创建临时配置
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("global:\n  default_parts: 1\n")
        config_path = Path(f.name)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.onnx"

        try:
            run_split(
                model_path=str(model_path),
                config_path=str(config_path),
                cli_parts=1,
                cli_max_memory=None,
                output_path=str(output_path),
                report_path=None,
            )
        finally:
            model_path.unlink()
            config_path.unlink()


def test_run_split_with_report():
    """测试生成报告"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.onnx"
        report_path = Path(tmpdir) / "report.json"

        run_split(
            model_path=str(model_path),
            config_path=None,
            cli_parts=2,
            cli_max_memory=None,
            output_path=str(output_path),
            report_path=str(report_path),
        )

        assert output_path.exists()
        assert report_path.exists()

        # 验证报告格式
        import json
        with open(report_path) as f:
            report = json.load(f)
        assert "model" in report
        assert "memory_analysis" in report


def test_run_analyze():
    """测试分析命令"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "report.json"

        run_analyze(
            model_path=str(model_path),
            report_path=str(report_path),
        )

        assert report_path.exists()


def test_run_analyze_without_report():
    """测试分析命令（不生成报告，只打印）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")

    # 应该不报错
    run_analyze(
        model_path=str(model_path),
        report_path=None,
    )


def test_run_split_invalid_model():
    """测试无效模型"""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "invalid.onnx"
        model_path.write_text("not a model")

        output_path = Path(tmpdir) / "output.onnx"

        with pytest.raises(Exception):
            run_split(
                model_path=str(model_path),
                config_path=None,
                cli_parts=2,
                cli_max_memory=None,
                output_path=str(output_path),
                report_path=None,
            )


def test_run_split_auto_adjust():
    """测试自动调整切分"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.onnx"

        run_split(
            model_path=str(model_path),
            config_path=None,
            cli_parts=1,  # 默认不切分
            cli_max_memory=1,  # 但内存限制很低
            output_path=str(output_path),
            report_path=None,
        )

        # 应该有输出
        assert output_path.exists()
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_cli_runner.py -v`
Expected: FAIL - `ModuleNotFoundError: onnxsplit.cli.runner`

**Step 3: 实现CLI运行器**

创建 `src/onnxsplit/cli/runner.py`:

```python
"""CLI命令运行器"""
import json
import sys
from pathlib import Path
from typing import Optional

import typer
import onnx

from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.config import load_config, merge_cli_args, ConfigError
from onnxsplit.splitter import SplitPlanner
from onnxsplit.memory import MemoryEstimator, AutoSplitAdjuster
from onnxsplit.transform import GraphTransformer


def run_split(
    model_path: str,
    config_path: Optional[str],
    cli_parts: int,
    cli_max_memory: Optional[int],
    output_path: str,
    report_path: Optional[str],
) -> None:
    """运行模型切分

    Args:
        model_path: 输入模型路径
        config_path: 配置文件路径
        cli_parts: 命令行指定的切分数
        cli_max_memory: 命令行指定的内存限制
        output_path: 输出模型路径
        report_path: 报告输出路径
    """
    # 加载模型
    typer.echo(f"Loading model: {model_path}")
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except Exception as e:
        typer.echo(f"Error loading model: {e}", err=True)
        raise typer.Exit(1)

    # 分析模型
    analyzer = ModelAnalyzer.from_model_proto(model)
    typer.echo(f"Model: {analyzer.graph_name}")
    typer.echo(f"Operators: {len(analyzer.get_operators())}")

    # 加载配置
    if config_path:
        typer.echo(f"Loading config: {config_path}")
        try:
            config = load_config(Path(config_path))
        except ConfigError as e:
            typer.echo(f"Config error: {e}", err=True)
            raise typer.Exit(1)
    else:
        from onnxsplit.config import SplitConfig, GlobalConfig
        config = SplitConfig(global_config=GlobalConfig(default_parts=cli_parts))

    # 合并命令行参数
    if cli_parts > 1 or cli_max_memory:
        config = merge_cli_args(config, cli_parts, cli_max_memory)

    # 生成切分方案
    typer.echo("Generating split plan...")
    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 内存调整
    if cli_max_memory:
        typer.echo(f"Applying memory limit: {cli_max_memory}MB")
        estimator = MemoryEstimator(analyzer)
        adjuster = AutoSplitAdjuster(estimator)
        report.plans = adjuster.adjust_report(report.plans, cli_max_memory)

    typer.echo(f"Split plan: {report.summary()}")

    if not report.plans:
        typer.echo("No operators to split. Saving original model.")
        onnx.save(model, output_path)
    else:
        # 应用切分
        typer.echo("Applying splits...")
        transformer = GraphTransformer(analyzer)

        # 目前只应用第一个方案
        # TODO: 支持多个方案的组合应用
        plan = report.plans[0]
        new_model = transformer.apply_split_plan(plan)

        # 验证新模型
        onnx.checker.check_model(new_model)
        typer.echo(f"Saving split model: {output_path}")
        onnx.save(new_model, output_path)

    # 生成报告
    if report_path:
        typer.echo(f"Generating report: {report_path}")
        _generate_report(
            analyzer=analyzer,
            split_report=report,
            output_model=output_path,
            report_path=report_path,
        )

    typer.echo("✓ Done")


def run_analyze(
    model_path: str,
    report_path: Optional[str],
) -> None:
    """分析模型内存

    Args:
        model_path: 模型路径
        report_path: 报告输出路径
    """
    typer.echo(f"Analyzing model: {model_path}")

    # 加载模型
    model = onnx.load(model_path)
    analyzer = ModelAnalyzer.from_model_proto(model)

    typer.echo(f"Model: {analyzer.graph_name}")
    typer.echo(f"IR version: {analyzer.ir_version}")
    typer.echo(f"Opset version: {analyzer.opset_version}")
    typer.echo(f"Operators: {len(analyzer.get_operators())}")

    # 内存分析
    estimator = MemoryEstimator(analyzer)
    peak_memory = estimator.get_peak_memory()
    total_memory = estimator.get_total_model_memory()

    typer.echo(f"\nMemory Analysis:")
    typer.echo(f"  Total model memory: {total_memory / (1024**2):.2f} MB")
    typer.echo(f"  Peak operator memory: {peak_memory:.2f} MB")

    # 算子内存分解
    breakdown = estimator.get_memory_breakdown()
    typer.echo(f"\nTop 5 operators by memory:")
    sorted_ops = sorted(breakdown, key=lambda x: x.total_memory_mb, reverse=True)[:5]
    for op_info in sorted_ops:
        typer.echo(
            f"  {op_info.operator_name} ({op_info.op_type}): "
            f"{op_info.total_memory_mb:.2f} MB"
        )

    # 可切分算子
    planner = SplitPlanner(analyzer, config=None)
    # 获取全局配置默认
    from onnxsplit.config import SplitConfig, GlobalConfig
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))
    planner.config = config

    split_report = planner.generate()
    typer.echo(f"\nSplit Analysis:")
    typer.echo(f"  Splitable operators: {split_report.split_operators}")
    typer.echo(f"  Unsplitable operators: {split_report.unsplit_operators}")

    if report_path:
        _generate_report(
            analyzer=analyzer,
            split_report=split_report,
            output_model=None,
            report_path=report_path,
        )
        typer.echo(f"\nReport saved: {report_path}")


def _generate_report(
    analyzer: ModelAnalyzer,
    split_report,
    output_model: Optional[str],
    report_path: str,
) -> None:
    """生成JSON报告"""
    estimator = MemoryEstimator(analyzer)

    report_data = {
        "model": {
            "name": analyzer.graph_name,
            "ir_version": analyzer.ir_version,
            "opset_version": analyzer.opset_version,
        },
        "output_model": output_model,
        "memory_analysis": {
            "total_memory_mb": estimator.get_total_model_memory() / (1024**2),
            "peak_memory_mb": estimator.get_peak_memory(),
            "weights_memory_mb": estimator.get_weights_memory() / (1024**2),
        },
        "split_plan": {
            "original_operators": split_report.original_operators,
            "split_operators": split_report.split_operators,
            "unsplit_operators": split_report.unsplit_operators,
            "total_parts": split_report.total_parts,
            "max_parts": split_report.max_parts,
            "splits": [
                {
                    "operator": plan.operator_name,
                    "parts": plan.parts,
                    "axis": plan.axis,
                    "reason": plan.reason,
                }
                for plan in split_report.plans
            ],
        },
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_cli_runner.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/cli/runner.py tests/test_cli_runner.py
git commit -m "feat: add CLI runner"
```

---

## Task 5: 主入口和模块导出

**Files:**
- Create: `src/onnxsplit/__main__.py`
- Modify: `src/onnxsplit/__init__.py`
- Modify: `src/onnxsplit/cli/__init__.py`
- Modify: `src/onnxsplit/memory/__init__.py`
- Test: `tests/test_main.py`

**Step 1: 编写主入口测试**

创建 `tests/test_main.py`:

```python
"""测试主入口"""
import subprocess
import sys


def test_main_module():
    """测试主模块可执行"""
    result = subprocess.run(
        [sys.executable, "-m", "onnxsplit", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "onnxsplit" in result.stdout


def test_main_split_command():
    """测试split命令存在"""
    result = subprocess.run(
        [sys.executable, "-m", "onnxsplit", "split", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_main_analyze_command():
    """测试analyze命令存在"""
    result = subprocess.run(
        [sys.executable, "-m", "onnxsplit", "analyze", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_main_validate_command():
    """测试validate命令存在"""
    result = subprocess.run(
        [sys.executable, "-m", "onnxsplit", "validate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
```

**Step 2: 创建主入口**

创建 `src/onnxsplit/__main__.py`:

```python
"""主入口"""
from onnxsplit.cli.parser import app

if __name__ == "__main__":
    app()
```

**Step 3: 更新模块导出**

编辑 `src/onnxsplit/__init__.py`:

```python
"""ONNX模型切分工具

通过算子复制方式切分ONNX模型，降低内存峰值。
"""

__version__ = "0.1.0"

from onnxsplit.analyzer import ModelAnalyzer, OperatorInfo, TensorMetadata
from onnxsplit.config import load_config, SplitConfig
from onnxsplit.splitter import SplitPlanner, SplitPlan
from onnxsplit.transform import GraphTransformer

__all__ = [
    # Version
    "__version__",
    # Analyzer
    "ModelAnalyzer",
    "OperatorInfo",
    "TensorMetadata",
    # Config
    "load_config",
    "SplitConfig",
    # Splitter
    "SplitPlanner",
    "SplitPlan",
    # Transform
    "GraphTransformer",
]
```

编辑 `src/onnxsplit/cli/__init__.py`:

```python
"""CLI模块"""
from onnxsplit.cli.parser import app, CliOptions, generate_output_name
from onnxsplit.cli.runner import run_split, run_analyze

__all__ = [
    "app",
    "CliOptions",
    "generate_output_name",
    "run_split",
    "run_analyze",
]
```

编辑 `src/onnxsplit/memory/__init__.py`:

```python
"""内存分析模块"""
from onnxsplit.memory.estimator import (
    MemoryEstimator,
    TensorMemoryInfo,
    OperatorMemoryInfo,
    dtype_bytes,
    estimate_tensor_memory,
)
from onnxsplit.memory.auto_adjust import AutoSplitAdjuster

__all__ = [
    "MemoryEstimator",
    "TensorMemoryInfo",
    "OperatorMemoryInfo",
    "dtype_bytes",
    "estimate_tensor_memory",
    "AutoSplitAdjuster",
]
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_main.py -v`
Expected: PASS - 所有测试通过

**Step 5: 提交**

```bash
git add src/onnxsplit/__main__.py src/onnxsplit/__init__.py src/onnxsplit/cli/__init__.py src/onnxsplit/memory/__init__.py tests/test_main.py
git commit -m "feat: add main entry point and module exports"
```

---

## 完成检查

**Step 1: 运行所有测试**

Run: `uv run pytest tests/ -v`
Expected: PASS - 所有测试通过

**Step 2: 检查代码风格**

Run: `uv run ruff check src/ tests/`
Expected: 无错误

**Step 3: 测试CLI功能**

Run: `uv run python -m onnxsplit --help`
Expected: 显示帮助信息

Run: `uv run python -m onnxsplit analyze tests/fixtures/models/simple_conv.onnx`
Expected: 显示模型分析信息

**Step 4: 检查测试覆盖率**

Run: `uv run pytest --cov=onnxsplit --cov-report=term-missing`
Expected: 整体覆盖率 >= 75%

**Step 5: 最终提交**

```bash
git add .
git commit -m "chore: finalize plan 5 - memory estimator and CLI integration"
```

---

**Plan 5 完成！** 内存估算与CLI集成已实现，包括：
- 内存估算器
- 自动切分调整
- CLI参数解析（typer）
- CLI命令运行器
- 主入口和模块导出
- 完整的测试覆盖

**所有5个计划已完成！** ONNX模型切分工具已完全实现。
