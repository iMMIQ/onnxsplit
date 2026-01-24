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
