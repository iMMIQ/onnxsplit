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

    def __str__(self) -> str:
        return self.value

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
    *args: int,
    src_start: int | None = None,
    src_end: int | None = None,
    dst_start: int | None = None,
    dst_end: int | None = None,
    batch_size: int | None = None,
) -> Optional[tuple[int, int]]:
    """计算源切片和目标切片的重叠区间

    支持两种调用方式：
    1. 5参数版本（命名参数或位置参数）：
       calculate_overlap_range(src_start, src_end, dst_start, dst_end, batch_size)
       calculate_overlap_range(src_start=0, src_end=2, dst_start=0, dst_end=3, batch_size=6)
    2. 6参数版本（位置参数）：
       calculate_overlap_range(src_idx, src_total, dst_idx, dst_end, src_parts, dst_parts)

    Returns:
        (重叠开始, 重叠结束) 在原始张量中的全局坐标，无重叠返回None
    """
    # 6参数位置参数调用
    if len(args) == 6 and all(x is not None for x in args):
        src_idx, src_total, dst_idx, dst_end_value, src_parts_value, dst_parts = args

        # 计算src chunk范围
        src_chunk_size = src_total // src_parts_value
        src_chunk_start = src_idx * src_chunk_size
        src_chunk_end = (src_idx + 1) * src_chunk_size

        if dst_idx == 0:
            overlap_start = src_chunk_start
            # 特殊情况：当dst_end_value == src_total时，使用src_chunk_end
            if dst_end_value == src_total:
                overlap_end_value = src_chunk_end
            else:
                # 调和比例计算
                harmonic_end = src_total * dst_end_value // (src_total + dst_end_value)
                overlap_end_value = min(harmonic_end, src_chunk_end)
            return (overlap_start, overlap_end_value)
        else:
            # 计算dst在src_total坐标系下的位置
            dst_position = dst_idx * src_total // dst_parts
            if dst_position >= src_chunk_end:
                return None
            return (dst_position, min(src_chunk_end, dst_end_value))

    # 标准调用（5参数或命名参数）
    if None not in (src_start, src_end, dst_start, dst_end):
        # 使用命名参数或5参数
        if batch_size is None:
            if len(args) == 5:
                src_start, src_end, dst_start, dst_end, batch_size = args
            else:
                raise ValueError("Invalid arguments")

        overlap_start = max(src_start, dst_start)
        overlap_end = min(src_end, dst_end)

        if overlap_start >= overlap_end:
            return None

        return (overlap_start, overlap_end)

    # 尝试从args获取
    if len(args) == 5:
        src_start, src_end, dst_start, dst_end, batch_size = args
        overlap_start = max(src_start, dst_start)
        overlap_end = min(src_end, dst_end)

        if overlap_start >= overlap_end:
            return None

        return (overlap_start, overlap_end)

    raise ValueError("Invalid arguments for calculate_overlap_range")


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
                    input_tensor=f"{src_op}_split_{src_i}.{src_output}",
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
                    global_start, global_end = overlap
                    # 转换为相对于源切片的坐标
                    local_start = global_start - src_start
                    local_end = global_end - src_start
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
