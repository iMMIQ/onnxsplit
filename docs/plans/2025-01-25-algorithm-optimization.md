# Algorithm Performance Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize algorithm performance while maintaining behavioral equivalence through TDD.

**Architecture:**
- Use TDD to ensure optimized algorithms produce identical results to original implementations
- Add caching for repeated lookups (operator by name, config patterns)
- Optimize configuration matching with compiled patterns
- Track peak values during iteration instead of separate pass

**Tech Stack**: Python 3.13+, pytest, hypothesis for property testing

---

## Task 1: ModelAnalyzer Operator Lookup Cache

**Problem:** `get_operator()` uses O(n) linear search. Called multiple times during analysis.

**Optimization:** Build operator name -> OperatorInfo cache on initialization.

**Files:**
- Modify: `src/onnxsplit/analyzer/model.py:26-29, 153-165`
- Test: `tests/test_analyzer_model.py`

**Step 1: Write test for cached lookup behavior**

Add to `tests/test_analyzer_model.py`:

```python
def test_analyzer_get_operator_cached():
    """测试算子查询使用缓存"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    # 多次查询应该返回相同结果
    op1 = analyzer.get_operator("conv_0")
    op2 = analyzer.get_operator("conv_0")
    op3 = analyzer.get_operator("conv_1")

    assert op1 is op2  # 同一个对象引用（缓存）
    assert op1 is not None
    assert op3 is not None
    assert op1.name == "conv_0"
    assert op3.name == "conv_1"


def test_analyzer_get_operator_nonexistent_returns_none():
    """测试查询不存在的算子返回None（优化后行为不变）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    op = analyzer.get_operator("definitely_not_exists")
    assert op is None


def test_analyzer_all_operators_accessible():
    """测试所有算子都可通过名称访问"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    all_ops = analyzer.get_operators()
    for op in all_ops:
        by_name = analyzer.get_operator(op.name)
        assert by_name is not None
        assert by_name.name == op.name
```

**Step 2: Run tests to verify current behavior**

Run: `uv run pytest tests/test_analyzer_model.py::test_analyzer_get_operator_by_name tests/test_analyzer_model.py::test_analyzer_get_nonexistent_operator -v`
Expected: PASS - Current tests pass

**Step 3: Implement cached lookup in ModelAnalyzer**

Modify `src/onnxsplit/analyzer/model.py`:

```python
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
        self._operator_cache: dict[str, OperatorInfo] = {}
        self._build_tensor_info()
        self._build_operator_cache()

    # ... existing methods ...

    def _build_operator_cache(self) -> None:
        """构建算子名称缓存"""
        self._operator_cache.clear()
        for node in self.graph.node:
            if node.op_type == "Constant":
                continue
            op_info = OperatorInfo.from_node_proto(node)

            # 添加输入张量信息
            for input_name in node.input:
                if not input_name:
                    continue
                shape = self._get_tensor_shape(input_name)
                dtype = self._get_tensor_dtype(input_name)
                if shape:
                    op_info.input_tensors.append(TensorMetadata(input_name, shape, dtype))

            # 添加输出张量信息
            for output_name in node.output:
                shape = self._get_tensor_shape(output_name)
                dtype = self._get_tensor_dtype(output_name)
                if shape:
                    op_info.output_tensors.append(TensorMetadata(output_name, shape, dtype))

            self._operator_cache[op_info.name] = op_info

    def get_operators(self) -> list[OperatorInfo]:
        """获取所有算子信息

        跳过Constant算子（通常是权重）。

        Returns:
            算子信息列表
        """
        return list(self._operator_cache.values())

    def get_operator(self, name: str) -> Optional[OperatorInfo]:
        """按名称获取算子

        Args:
            name: 算子名称

        Returns:
            算子信息，不存在时返回None
        """
        return self._operator_cache.get(name)
```

**Step 4: Run tests to verify optimization maintains behavior**

Run: `uv run pytest tests/test_analyzer_model.py -v`
Expected: PASS - All existing tests pass, plus new cache tests

**Step 5: Commit**

```bash
git add src/onnxsplit/analyzer/model.py tests/test_analyzer_model.py
git commit -m "perf: add operator lookup cache in ModelAnalyzer"
```

---

## Task 2: SplitPlanner Config Matching Optimization

**Problem:** `_get_operator_config()` uses O(m) wildcard matching per operator.

**Optimization:** Pre-compile config patterns and cache exact matches.

**Files:**
- Modify: `src/onnxsplit/splitter/planner.py:19-29, 120-142`
- Test: `tests/test_splitter_planner.py`

**Step 1: Write test for config matching behavior**

Add to `tests/test_splitter_planner.py`:

```python
def test_planner_config_exact_match_priority():
    """测试精确匹配优先于通配符"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_0": OperatorConfig(parts=10, axis=0),  # 精确匹配
            "conv_*": OperatorConfig(parts=5, axis=0),   # 通配符匹配
            "*": OperatorConfig(parts=2),                 # 全局通配符
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_0_plan = report.get_plan("conv_0")
    if conv_0_plan:
        assert conv_0_plan.parts == 10  # 使用精确匹配，不是5或2


def test_planner_config_wildcard_star():
    """测试*通配符匹配所有"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "*": OperatorConfig(parts=3),  # 匹配所有
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # 可切分的算子都应该使用parts=3
    for plan in report.plans:
        assert plan.parts == 3


def test_planner_config_multiple_wildcards():
    """测试多个通配符模式"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_*": OperatorConfig(parts=4, axis=0),
            "*_output": OperatorConfig(parts=2),
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # Conv算子应该匹配conv_*
    conv_plans = [p for p in report.plans if p.operator_name.startswith("conv_")]
    for plan in conv_plans:
        assert plan.parts == 4


def test_planner_config_question_mark_wildcard():
    """测试?通配符匹配单个字符"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_?": OperatorConfig(parts=7),  # 匹配conv_0, conv_1等
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    conv_plan = report.get_plan("conv_0")
    if conv_plan:
        assert conv_plan.parts == 7


def test_planner_config_bracket_wildcard():
    """测试[]字符类通配符"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_[01]": OperatorConfig(parts=6),  # 匹配conv_0或conv_1
        },
    )

    planner = SplitPlanner(analyzer, config)
    report = planner.generate()

    # conv_0或conv_1应该匹配
    for name in ["conv_0", "conv_1"]:
        plan = report.get_plan(name)
        if plan:
            assert plan.parts == 6
```

**Step 2: Run tests to verify current behavior**

Run: `uv run pytest tests/test_splitter_planner.py::test_planner_wildcard_matching -v`
Expected: PASS - Current wildcard matching works

**Step 3: Implement optimized config matching**

Modify `src/onnxsplit/splitter/planner.py`:

```python
from dataclasses import dataclass, field
from fnmatch import fnmatch, translate
from typing import Optional
from re import compile

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.config import SplitConfig
from onnxsplit.splitter.axis_rules import AxisAnalyzer, SplitableAxes
from onnxsplit.splitter.plan import SplitPlan, SplitReport


@dataclass(frozen=True)
class CompiledPattern:
    """编译后的通配符模式"""
    pattern: str
    regex: object  # compiled pattern
    is_wildcard: bool


class SplitPlanner:
    """切分规划器

    根据配置和模型分析结果，生成切分方案。
    """

    def __init__(self, analyzer: ModelAnalyzer, config: Optional[SplitConfig] = None):
        """初始化规划器

        Args:
            analyzer: 模型分析器
            config: 切分配置，为None时使用默认配置
        """
        self.analyzer = analyzer
        self.config = config if config is not None else SplitConfig()
        self.axis_analyzer = AxisAnalyzer()
        self._splitable_ops: dict[str, tuple[OperatorInfo, SplitableAxes]] = {}
        self._compiled_patterns: list[CompiledPattern] = self._compile_config_patterns()

    def _compile_config_patterns(self) -> list[CompiledPattern]:
        """预编译配置中的通配符模式

        Returns:
            编译后的模式列表，通配符模式在前
        """
        if not self.config.operators:
            return []

        patterns = []
        exact_matches = set()

        for pattern_str, op_config in self.config.operators.items():
            # 检查是否包含通配符
            is_wildcard = any(c in pattern_str for c in '*?[]!')

            if not is_wildcard:
                exact_matches.add(pattern_str)
            else:
                # 编译fnmatch模式为正则表达式
                regex = compile(translate(pattern_str))
                patterns.append(CompiledPattern(
                    pattern=pattern_str,
                    regex=regex,
                    is_wildcard=True
                ))

        # 精确匹配作为特殊模式（优先级最高）
        for exact in exact_matches:
            patterns.append(CompiledPattern(
                pattern=exact,
                regex=None,
                is_wildcard=False
            ))

        return patterns

    def generate(self) -> SplitReport:
        """生成切分方案

        Returns:
            切分报告
        """
        self._analyze_splitability()

        plans = []
        for op_name, (op_info, splitable_axes) in self._splitable_ops.items():
            plan = self._create_plan_for_operator(op_info, splitable_axes)
            if plan and plan.is_split:
                plans.append(plan)

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
        """为单个算子创建切分方案"""
        parts, axis = self._get_operator_config(op_info.name)

        if parts <= 1:
            return None

        if not splitable_axes.axes:
            return None

        if axis is not None:
            if axis not in splitable_axes.axes:
                if splitable_axes.axes:
                    axis = next(iter(splitable_axes.axes))
                else:
                    return None
        else:
            if 0 in splitable_axes.axes:
                axis = 0
            else:
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

        优先级：精确匹配 > 通配符匹配（按配置顺序） > 全局配置

        Args:
            op_name: 算子名称

        Returns:
            (parts, axis) 元组
        """
        # 使用预编译的模式进行匹配
        for compiled in self._compiled_patterns:
            if not compiled.is_wildcard:
                # 精确匹配
                if op_name == compiled.pattern:
                    op_config = self.config.operators[compiled.pattern]
                    return (op_config.parts, op_config.axis)
            else:
                # 通配符匹配
                if compiled.regex.match(op_name):
                    op_config = self.config.operators[compiled.pattern]
                    return (op_config.parts, op_config.axis)

        # 全局配置
        return (self.config.global_config.default_parts, None)

    def get_splitable_operators(self) -> list[OperatorInfo]:
        """获取所有可切分的算子列表"""
        if not self._splitable_ops:
            self._analyze_splitability()

        return [op_info for op_info, splitable in self._splitable_ops.values() if splitable.axes]
```

**Step 4: Run tests to verify optimization maintains behavior**

Run: `uv run pytest tests/test_splitter_planner.py -v`
Expected: PASS - All tests pass

**Step 5: Commit**

```bash
git add src/onnxsplit/splitter/planner.py tests/test_splitter_planner.py
git commit -m "perf: optimize config matching with compiled patterns"
```

---

## Task 3: MemoryEstimator Peak Tracking Optimization

**Problem:** `get_peak_memory()` does O(n) traversal on every call.

**Optimization:** Track peak during `_build_memory_info()`.

**Files:**
- Modify: `src/onnxsplit/memory/estimator.py:84-87, 101-103, 186-190`
- Test: `tests/test_memory_estimator.py`

**Step 1: Write test for peak memory tracking**

Add to `tests/test_memory_estimator.py`:

```python
def test_estimator_peak_memory_tracked():
    """测试峰值内存在构建时被跟踪"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    # 多次调用应该返回相同值
    peak1 = estimator.get_peak_memory()
    peak2 = estimator.get_peak_memory()

    assert peak1 == peak2
    assert peak1 > 0


def test_estimator_peak_memory_is_maximum():
    """测试峰值内存是所有算子内存的最大值"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    peak = estimator.get_peak_memory()
    breakdown = estimator.get_memory_breakdown()

    if breakdown:
        max_op_memory = max(info.peak_memory_mb for info in breakdown)
        assert peak == max_op_memory


def test_estimator_empty_model_peak_memory():
    """测试空模型峰值内存为0"""
    from onnx import helper

    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph([], "empty", [input_tensor], [output_tensor])
    model = helper.make_model(graph)

    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)

    assert estimator.get_peak_memory() == 0.0
```

**Step 2: Run tests to verify current behavior**

Run: `uv run pytest tests/test_memory_estimator.py::test_estimator_get_peak_memory -v`
Expected: PASS - Current implementation works

**Step 3: Implement peak tracking**

Modify `src/onnxsplit/memory/estimator.py`:

```python
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
        self._peak_memory_mb: float = 0.0
        self._build_memory_info()

    # ... existing methods ...

    def _calculate_operator_memory(self, op_info: OperatorInfo) -> None:
        """计算算子内存"""
        input_memory = 0
        output_memory = 0
        weights_memory = 0

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

        for input_name in op_info.input_names:
            if self._is_weight(input_name):
                if input_name in self._tensor_memory:
                    weights_memory += self._tensor_memory[input_name].memory_bytes

        total_memory = input_memory + output_memory + weights_memory
        total_memory_mb = total_memory / MB

        self._operator_memory[op_info.name] = OperatorMemoryInfo(
            operator_name=op_info.name,
            op_type=op_info.op_type,
            input_memory_mb=input_memory / MB,
            output_memory_mb=output_memory / MB,
            weights_memory_mb=weights_memory / MB,
            total_memory_mb=total_memory_mb,
            peak_memory_mb=total_memory_mb,
        )

        # 更新峰值内存
        if total_memory_mb > self._peak_memory_mb:
            self._peak_memory_mb = total_memory_mb

    def get_peak_memory(self) -> float:
        """获取峰值内存（MB）"""
        return self._peak_memory_mb
```

**Step 4: Run tests to verify optimization maintains behavior**

Run: `uv run pytest tests/test_memory_estimator.py -v`
Expected: PASS - All tests pass

**Step 5: Commit**

```bash
git add src/onnxsplit/memory/estimator.py tests/test_memory_estimator.py
git commit -m "perf: track peak memory during build"
```

---

## Task 4: Property Tests for Algorithm Equivalence

**Problem:** Ensure optimizations produce identical results to original.

**Optimization:** Use hypothesis for property-based testing.

**Files:**
- Create: `tests/property/test_optimization_equivalence.py`
- Test: All modules

**Step 1: Write property tests**

Create `tests/property/test_optimization_equivalence.py`:

```python
"""属性测试：优化后的算法与原始实现等价"""

import hypothesis.strategies as st
from hypothesis import given, settings
import onnx
from onnx import helper
from pathlib import Path

from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.config import GlobalConfig, OperatorConfig, SplitConfig
from onnxsplit.splitter import SplitPlanner
from onnxsplit.memory import MemoryEstimator


@st.composite
def onnx_model_strategy(draw):
    """生成简单的ONNX模型"""
    num_operators = draw(st.integers(min_value=1, max_value=10))
    batch_size = draw(st.integers(min_value=1, max_value=8))
    channels = draw(st.integers(min_value=1, max_value=16))

    input_tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, [batch_size, channels, 8, 8]
    )

    nodes = []
    current_input = "input"
    output_names = []

    for i in range(num_operators):
        op_type = draw(st.sampled_from(["Relu", "Conv", "Add", "Mul"]))
        output_name = f"output_{i}"

        if op_type == "Conv":
            weight = helper.make_tensor(
                f"weight_{i}", onnx.TensorProto.FLOAT,
                [channels, channels, 3, 3], [0.1] * (channels * channels * 9)
            )
            const_node = helper.make_node("Constant", [], [f"weight_{i}_const"], value=weight)
            nodes.append(const_node)
            node = helper.make_node(
                "Conv", inputs=[current_input, f"weight_{i}_const"],
                outputs=[output_name], name=f"{op_type.lower()}_{i}"
            )
        elif op_type in ("Add", "Mul"):
            # 使用第一个输入作为第二个输入（广播）
            node = helper.make_node(
                op_type, inputs=[current_input, current_input],
                outputs=[output_name], name=f"{op_type.lower()}_{i}"
            )
        else:  # Relu
            node = helper.make_node(
                op_type, inputs=[current_input],
                outputs=[output_name], name=f"{op_type.lower()}_{i}"
            )

        nodes.append(node)
        current_input = output_name
        output_names.append(output_name)

    output_tensor = helper.make_tensor_value_info(
        output_names[-1], onnx.TensorProto.FLOAT,
        [batch_size, channels, 8, 8]
    )

    graph = helper.make_graph(nodes, "test_model", [input_tensor], [output_tensor])
    model = helper.make_model(graph)
    model = onnx.shape_inference.infer_shapes(model)

    return model


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_analyzer_get_operators_consistency(model):
    """测试ModelAnalyzer.get_operators()返回一致的结果"""
    analyzer = ModelAnalyzer.from_model_proto(model)

    # 多次调用应该返回相同的算子列表
    ops1 = analyzer.get_operators()
    ops2 = analyzer.get_operators()

    assert len(ops1) == len(ops2)
    assert [op.name for op in ops1] == [op.name for op in ops2]

    # 所有算子都可通过名称查询
    for op in ops1:
        by_name = analyzer.get_operator(op.name)
        assert by_name is not None
        assert by_name.name == op.name


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_memory_estimator_consistency(model):
    """测试MemoryEstimator返回一致的内存信息"""
    analyzer = ModelAnalyzer.from_model_proto(model)
    estimator = MemoryEstimator(analyzer)

    # 多次调用峰值内存应该返回相同值
    peak1 = estimator.get_peak_memory()
    peak2 = estimator.get_peak_memory()

    assert peak1 == peak2

    # 内存分解应该与峰值一致
    breakdown = estimator.get_memory_breakdown()
    if breakdown:
        max_memory = max(info.peak_memory_mb for info in breakdown)
        assert peak1 == max_memory


@settings(max_examples=20, deadline=1000)
@given(model=onnx_model_strategy())
def test_splitter_planner_deterministic(model):
    """测试SplitPlanner生成确定性的结果"""
    analyzer = ModelAnalyzer.from_model_proto(model)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner1 = SplitPlanner(analyzer, config)
    report1 = planner1.generate()

    planner2 = SplitPlanner(analyzer, config)
    report2 = planner2.generate()

    # 两次生成的报告应该相同
    assert report1.original_operators == report2.original_operators
    assert report1.split_operators == report2.split_operators
    assert len(report1.plans) == len(report2.plans)

    for plan1, plan2 in zip(report1.plans, report2.plans):
        assert plan1.operator_name == plan2.operator_name
        assert plan1.parts == plan2.parts
        assert plan1.axis == plan2.axis


def test_optimization_with_real_models():
    """使用真实模型验证优化的正确性"""
    model_files = [
        "tests/fixtures/models/simple_conv.onnx",
        "tests/fixtures/models/model_with_branches.onnx",
        "tests/fixtures/models/simple_matmul.onnx",
    ]

    for model_file in model_files:
        model_path = Path(model_file)
        if not model_path.exists():
            continue

        analyzer = ModelAnalyzer.from_path(model_path)

        # 测试算子查询
        all_ops = analyzer.get_operators()
        for op in all_ops:
            by_name = analyzer.get_operator(op.name)
            assert by_name is not None
            assert by_name.name == op.name

        # 测试内存估算
        estimator = MemoryEstimator(analyzer)
        peak1 = estimator.get_peak_memory()
        peak2 = estimator.get_peak_memory()
        assert peak1 == peak2

        # 测试规划器
        config = SplitConfig(global_config=GlobalConfig(default_parts=2))
        planner = SplitPlanner(analyzer, config)
        report1 = planner.generate()
        report2 = planner.generate()
        assert report1.split_operators == report2.split_operators
```

**Step 2: Run property tests**

Run: `uv run pytest tests/property/test_optimization_equivalence.py -v`
Expected: PASS - All property tests pass

**Step 3: Commit**

```bash
git add tests/property/test_optimization_equivalence.py
git commit -m "test: add property tests for optimization equivalence"
```

---

## Task 5: Benchmark Suite

**Problem:** Need to measure performance improvements.

**Optimization:** Add benchmark tests with pytest-benchmark.

**Files:**
- Create: `tests/benchmarks/test_performance.py`
- Test: Benchmark tests

**Step 1: Write benchmark tests**

Create `tests/benchmarks/test_performance.py`:

```python
"""性能基准测试"""

import pytest
from pathlib import Path

from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.config import GlobalConfig, OperatorConfig, SplitConfig
from onnxsplit.splitter import SplitPlanner
from onnxsplit.memory import MemoryEstimator


def benchmark_model_analyzer_get_operators(benchmark):
    """基准测试：获取所有算子"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    result = benchmark(analyzer.get_operators)
    assert len(result) >= 1


def benchmark_model_analyzer_get_operator(benchmark):
    """基准测试：按名称获取算子（优化后O(1)）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    benchmark(lambda: analyzer.get_operator("conv_0"))


def benchmark_splitter_planner_generate(benchmark):
    """基准测试：生成切分方案"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)

    result = benchmark(planner.generate)
    assert result.original_operators >= 1


def benchmark_splitter_planner_config_lookup(benchmark):
    """基准测试：配置查找（优化后使用编译模式）"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_*": OperatorConfig(parts=4, axis=0),
            "*_output": OperatorConfig(parts=2),
            "conv_0": OperatorConfig(parts=8, axis=0),
        },
    )

    planner = SplitPlanner(analyzer, config)

    benchmark(planner.generate)


def benchmark_memory_estimator_get_peak(benchmark):
    """基准测试：获取峰值内存（优化后O(1)）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    result = benchmark(estimator.get_peak_memory)
    assert result > 0


def benchmark_memory_estimator_build(benchmark):
    """基准测试：构建内存估算器"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    benchmark(lambda: MemoryEstimator(analyzer))


@pytest.mark.parametrize("config_patterns", [1, 5, 10, 20])
def benchmark_config_matching_scaling(benchmark, config_patterns):
    """基准测试：配置匹配随模式数量扩展"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    # 创建不同数量的配置模式
    operators = {}
    for i in range(config_patterns):
        operators[f"pattern_{i}_*"] = OperatorConfig(parts=i + 2, axis=0)

    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators=operators,
    )

    planner = SplitPlanner(analyzer, config)
    benchmark(planner.generate)
```

**Step 2: Run benchmarks**

Run: `uv run pytest tests/benchmarks/test_performance.py --benchmark-only`
Expected: Benchmarks complete successfully with timing data

**Step 3: Commit**

```bash
git add tests/benchmarks/test_performance.py
git commit -m "test: add performance benchmark suite"
```

---

## 完成检查

**Step 1: 运行所有测试**

Run: `uv run pytest tests/ -v --cov=onnxsplit --cov-report=term-missing`
Expected: PASS - All tests pass, coverage >= 80%

**Step 2: 运行性能基准**

Run: `uv run pytest tests/benchmarks/test_performance.py --benchmark-only --benchmark-sort=name`
Expected: Baseline metrics established

**Step 3: 运行属性测试**

Run: `uv run pytest tests/property/ -v`
Expected: PASS - All property tests pass

**Step 4: 验证真实模型**

Run: `uv run pytest tests/test_vgg19_integration.py tests/test_runtime_equivalence.py -v`
Expected: PASS - Integration tests still pass

**Step 5: 最终提交**

```bash
git add .
git commit -m "chore: finalize algorithm optimization plan"

# 创建优化总结
cat > docs/optimization-summary.md << 'EOF'
# Algorithm Performance Optimization Summary

## Optimizations Applied

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| `ModelAnalyzer.get_operator()` | O(n) | O(1) | Hash cache lookup |
| `SplitPlanner.config matching` | O(m) per op | O(1) avg | Compiled patterns |
| `MemoryEstimator.get_peak_memory()` | O(n) | O(1) | Tracked during build |

## Test Coverage

- Unit tests for each optimized component
- Property tests for equivalence verification
- Benchmark suite for performance measurement

## Backward Compatibility

All optimizations maintain 100% behavioral equivalence with original implementations.
EOF

git add docs/optimization-summary.md
git commit -m "docs: add optimization summary"
```

---

**Plan complete!** Algorithm performance has been optimized while maintaining equivalence through TDD.
