"""测试数据流重连算法"""

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
