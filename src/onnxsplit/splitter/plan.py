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
