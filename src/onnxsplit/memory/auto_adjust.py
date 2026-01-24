"""自动切分调整"""

from onnxsplit.memory.estimator import MemoryEstimator
from onnxsplit.splitter.axis_rules import AxisAnalyzer
from onnxsplit.splitter.plan import SplitPlan


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
        max_parts_search = self.max_parts

        # 快速检查
        if total_memory_mb / min_parts <= max_memory_mb:
            return min_parts

        # 二分查找
        while min_parts < max_parts_search:
            mid_parts = (min_parts + max_parts_search) // 2
            per_part = total_memory_mb / mid_parts

            if per_part <= max_memory_mb:
                max_parts_search = mid_parts
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

        return [self.adjust_plan(plan, max_memory_mb) for plan in plans]
