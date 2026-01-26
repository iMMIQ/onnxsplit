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
        min_parts: int = 1,
    ) -> SplitPlan:
        """调整切分方案

        Args:
            plan: 原始切分方案
            max_memory_mb: 内存限制（MB），None表示不限制
            min_parts: 最小切分数限制（来自 CLI -p 参数）

        Returns:
            调整后的切分方案
        """
        # 修复Bug 1: 移除 `not plan.is_split` 检查
        # 当 parts=1 时，is_split=False，但仍需要根据内存限制进行调整
        if max_memory_mb is None and min_parts <= 1:
            return plan

        # 如果 axis 为 None，无法切分
        if plan.axis is None:
            return plan

        # 获取算子信息
        op_info = self.estimator.analyzer.get_operator(plan.operator_name)
        if op_info is None:
            return plan

        # 获取内存信息
        op_mem = self.estimator.get_operator_memory(op_info)
        if op_mem is None or op_mem.total_memory_mb == 0:
            return plan

        # 应用最小切分数限制
        base_parts = max(plan.parts, min_parts)

        # 修复Bug 2: 首先验证当前 parts 是否有效（能整除维度）
        # 即使不需要内存调整，无效的 parts 也应该被修正
        validated_parts = self._validate_and_adjust_parts(
            op_info, plan.axis, base_parts
        )

        # 检查验证后的 parts 是否真的有效（能整除维度）
        if not self._is_parts_valid(op_info, plan.axis, validated_parts):
            # 无法找到有效的切分数，放弃切分
            # 返回 parts=1 的方案（不切分）
            return SplitPlan(
                operator_name=plan.operator_name,
                parts=1,
                axis=plan.axis,
                slice_ranges=plan.slice_ranges,
                reason=f"Cannot split {base_parts} parts - dimension not divisible, using 1 (no split)",
            )

        # 如果没有内存限制，只返回验证后的 parts
        if max_memory_mb is None:
            if validated_parts != plan.parts:
                return SplitPlan(
                    operator_name=plan.operator_name,
                    parts=validated_parts,
                    axis=plan.axis,
                    slice_ranges=plan.slice_ranges,
                    reason=f"Adjusted from {plan.parts} to {validated_parts} for min_parts constraint",
                )
            return plan

        # 检查是否需要内存调整
        per_part_memory = op_mem.total_memory_mb / validated_parts
        if per_part_memory <= max_memory_mb:
            # 不需要内存调整，但可能修正了 parts
            if validated_parts != plan.parts:
                return SplitPlan(
                    operator_name=plan.operator_name,
                    parts=validated_parts,
                    axis=plan.axis,
                    slice_ranges=plan.slice_ranges,
                    reason=f"Adjusted from {plan.parts} to {validated_parts} for dimension divisibility",
                )
            return plan

        # 计算需要的切分数
        needed_parts = self._calculate_needed_parts(
            op_mem.total_memory_mb, max_memory_mb, validated_parts
        )

        # 限制在max_parts范围内
        final_parts = min(needed_parts, self.max_parts)

        # 再次验证（因为 _calculate_needed_parts 可能返回不能整除的值）
        final_parts = self._validate_and_adjust_parts(
            op_info, plan.axis, final_parts
        )

        # 再次检查验证后的 parts 是否有效
        if not self._is_parts_valid(op_info, plan.axis, final_parts):
            # 无法满足内存约束，放弃切分
            return SplitPlan(
                operator_name=plan.operator_name,
                parts=1,
                axis=plan.axis,
                slice_ranges=plan.slice_ranges,
                reason=f"Cannot satisfy memory constraint - dimension not divisible, using 1 (no split)",
            )

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

    def _is_parts_valid(
        self,
        op_info,
        axis: int,
        parts: int,
    ) -> bool:
        """检查切分数是否能整除目标维度

        Args:
            op_info: 算子信息
            axis: 切分轴
            parts: 切分数

        Returns:
            True 如果能整除所有相关维度，False 否则
        """
        # 检查是否是权重（包括 Constant 节点产生的）
        def _is_weight(tensor_name: str) -> bool:
            # 检查是否在 initializer 中
            if any(init.name == tensor_name
                   for init in self.estimator.analyzer.model.graph.initializer):
                return True
            # 检查是否由 Constant 节点产生
            for node in self.estimator.analyzer.model.graph.node:
                if node.op_type == "Constant" and tensor_name in node.output:
                    return True
            return False

        has_valid_dim = False  # 用于跟踪是否有至少一个有效维度

        for tensor in op_info.input_tensors:
            # 跳过权重
            if _is_weight(tensor.name):
                continue

            shape = tensor.shape
            if not shape or len(shape) <= axis:
                continue

            dim_size = shape[axis]
            if dim_size <= 0:
                # 动态维度，假设有效
                has_valid_dim = True
                continue

            # 对于非权重输入，检查是否能整除
            # 注意：广播输入（如 [8,8] 广播到 [18,8,8]）已经在上面被跳过
            # 因为它们通常是 Constant 节点的输出
            has_valid_dim = True

            # 检查是否能整除
            if dim_size % parts != 0:
                return False

        # 如果没有有效维度（全是动态维度或权重），返回True
        return True

    def _validate_and_adjust_parts(
        self,
        op_info,
        axis: int,
        parts: int,
    ) -> int:
        """验证并调整切分数，确保能整除目标维度

        如果计算出的 parts 不能整除目标维度，向上查找能整除的值。
        如果无法找到合适的切分数（如 parts > 维度大小），返回原值。

        Args:
            op_info: 算子信息
            axis: 切分轴
            parts: 初始计算的切分数

        Returns:
            验证后的切分数，如果不能整除则返回原值
        """
        # 检查是否是权重（包括 Constant 节点产生的）
        def _is_weight(tensor_name: str) -> bool:
            # 检查是否在 initializer 中
            if any(init.name == tensor_name
                   for init in self.estimator.analyzer.model.graph.initializer):
                return True
            # 检查是否由 Constant 节点产生
            for node in self.estimator.analyzer.model.graph.node:
                if node.op_type == "Constant" and tensor_name in node.output:
                    return True
            return False

        # 收集所有需要检查的维度大小（跳过权重）
        dim_sizes = []
        for tensor in op_info.input_tensors:
            # 跳过权重
            if _is_weight(tensor.name):
                continue

            shape = tensor.shape
            if not shape or len(shape) <= axis:
                continue

            dim_size = shape[axis]
            if dim_size <= 0:
                # 动态维度，无法验证
                continue

            dim_sizes.append(dim_size)

        if not dim_sizes:
            # 没有有效维度，使用原始值
            return parts

        # 检查所有维度，找到需要调整的
        max_adjusted_parts = parts
        can_split = True
        for dim_size in dim_sizes:
            if dim_size % parts != 0:
                # 不能整除，向上查找能整除的值
                adjusted, found = self._find_divisible_parts(dim_size, parts, dim_size)
                if found:
                    max_adjusted_parts = max(max_adjusted_parts, adjusted)
                else:
                    # 找不到合适的切分数，标记为不可切分
                    can_split = False
                    break

        # 如果无法找到合适的切分数，返回原值（调用者应检查是否能整除）
        return max_adjusted_parts if can_split else parts

    def _find_divisible_parts(
        self,
        dim_size: int,
        min_parts: int,
        max_dim: int,
    ) -> tuple[int, bool]:
        """查找能整除维度的最小切分数

        Args:
            dim_size: 目标维度大小
            min_parts: 最小切分数
            max_dim: 最大维度（用于计算搜索上限）

        Returns:
            (parts, found) 元组
            - parts: 找到的切分数（仅在 found=True 时有效）
            - found: 是否找到能整除的切分数
        """
        # 如果 min_parts 已经大于维度大小，无法切分
        if min_parts > dim_size:
            return (min_parts, False)

        # 搜索上限：min(维度大小, min_parts * 4, 256)
        search_limit = min(max_dim, min_parts * 4, self.max_parts)

        # 从 min_parts 开始向上查找
        for p in range(min_parts, search_limit + 1):
            if dim_size >= p and dim_size % p == 0:
                return (p, True)

        # 找不到合适的，返回原值并标记为未找到
        return (min_parts, False)

    def adjust_report(
        self,
        plans: list[SplitPlan],
        max_memory_mb: float | None,
        min_parts: int = 1,
    ) -> list[SplitPlan]:
        """批量调整切分方案

        Args:
            plans: 切分方案列表
            max_memory_mb: 内存限制
            min_parts: 最小切分数限制（来自 CLI -p 参数）

        Returns:
            调整后的方案列表
        """
        if max_memory_mb is None and min_parts <= 1:
            return plans

        return [self.adjust_plan(plan, max_memory_mb, min_parts) for plan in plans]
