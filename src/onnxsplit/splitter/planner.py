"""切分规划器"""

from dataclasses import dataclass
from fnmatch import translate
from re import compile
from typing import Optional, Pattern
from typing import TYPE_CHECKING

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.config import OperatorConfig, SplitConfig
from onnxsplit.splitter.axis_rules import AxisAnalyzer, SplitableAxes
from onnxsplit.splitter.plan import SplitPlan, SplitReport

@dataclass(frozen=True)
class CompiledPattern:
    """编译后的通配符模式

    Attributes:
        pattern: 原始模式字符串
        regex: 编译后的正则表达式
        is_wildcard: 是否为通配符模式
        config: 关联的算子配置
    """
    pattern: str
    regex: Pattern[str]
    is_wildcard: bool
    config: OperatorConfig


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
        self._warnings: list[str] = []
        self._invalid_axis_rule_warnings: set[tuple[str, str]] = set()
        self._matched_integer_axis_rules: dict[tuple[str, int], bool] = {}
        self._splitable_ops: dict[str, tuple[OperatorInfo, SplitableAxes]] = {}
        self._compiled_patterns: list[CompiledPattern] = []
        self._exact_match_patterns: dict[str, OperatorConfig] = {}
        self._compile_config_patterns()

    def generate(self) -> SplitReport:
        """生成切分方案

        Returns:
            切分报告
        """
        self._warnings.clear()
        self._invalid_axis_rule_warnings.clear()
        self._matched_integer_axis_rules.clear()

        # 分析所有算子的可切分性
        self._analyze_splitability()

        # 生成切分方案
        # 注意：即使 parts=1 (is_split=False)，也保留在 plans 中
        # 这样 AutoSplitAdjuster 可以根据内存限制进行调整
        plans = []
        for op_name, (op_info, splitable_axes) in self._splitable_ops.items():
            plan = self._create_plan_for_operator(op_info, splitable_axes)
            if plan:
                plans.append(plan)

        # 统计（只统计 is_split=True 的方案）
        total_ops = len(self._splitable_ops)
        split_ops = sum(1 for p in plans if p.is_split)
        unsplit_ops = total_ops - split_ops

        self._warn_never_usable_integer_axis_rules()

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
        candidate_axes, auto_planning_disabled = self._get_candidate_axes(
            op_info,
            splitable_axes,
            axis,
        )

        if auto_planning_disabled:
            return None

        # parts=1表示不切分，但仍创建方案以便 adjuster 根据内存限制调整
        if parts <= 1:
            # 创建 parts=1 的占位方案，标记为 is_split=False
            # 如果没有可切分轴，返回 None（无法调整）
            if not candidate_axes:
                return None
            return SplitPlan(
                operator_name=op_info.name,
                parts=1,
                axis=candidate_axes[0] if candidate_axes else 0,
                reason="No split requested (parts=1), but adjustable for memory limit",
            )

        # 检查可切分性
        if not candidate_axes:
            # 没有可切分轴
            return None

        # 尝试每个候选轴，直到找到可用的
        for try_axis in candidate_axes:
            found, adjusted_parts, warning = self._find_suitable_parts(op_info, try_axis, parts)

            if found:
                # 找到合适的切分方案
                if adjusted_parts != parts:
                    # 切分数被调整，添加信息日志（非警告）
                    # 可选：在 verbose 模式下输出
                    pass
                return SplitPlan(
                    operator_name=op_info.name,
                    parts=adjusted_parts,
                    axis=try_axis,
                    reason=splitable_axes.reason,
                )

        # 所有候选轴都无法切分
        self._add_warning(
            f"{op_info.name}: skipped split - tried axes {candidate_axes}, "
            f"none could be split into {parts} parts"
        )
        return None

    def _is_evenly_splittable(
        self,
        op_info: OperatorInfo,
        axis: int,
        parts: int,
    ) -> bool:
        """检查算子是否可以在指定轴上均匀分割

        Args:
            op_info: 算子信息
            axis: 切分轴
            parts: 切分份数

        Returns:
            True如果可以均匀分割，False否则
        """
        has_non_weight_input = False

        # 检查所有输入张量
        for tensor in op_info.input_tensors:
            # 跳过权重（常数）
            if self._is_weight(tensor.name):
                continue

            has_non_weight_input = True
            shape = tensor.shape
            # 检查形状是否有效
            if not shape or len(shape) <= axis:
                continue

            dim_size = shape[axis]

            # 动态维度（dim_value <= 0 或特殊值0）假设可以分割
            if dim_size <= 0:
                continue

            # 检查是否可以均匀分割
            if dim_size < parts or dim_size % parts != 0:
                return False

        # 如果没有非权重输入，不能分割
        return has_non_weight_input

    def _find_suitable_parts(
        self,
        op_info: OperatorInfo,
        axis: int,
        initial_parts: int,
    ) -> tuple[bool, int | None, str | None]:
        """查找适合的切分数

        当初始 parts 不能整除维度时，向上查找能整除的值。
        搜索上限：min(维度大小, initial_parts * 4, 256)

        Args:
            op_info: 算子信息
            axis: 切分轴
            initial_parts: 初始切分数

        Returns:
            (found, parts, warning_message)
            - found: 是否找到合适的切分数
            - parts: 找到的切分数（仅在 found=True 时有效）
            - warning_message: 警告信息（仅在 found=False 时有效）
        """
        # 收集所有需要检查的维度大小
        dim_sizes = []
        for tensor in op_info.input_tensors:
            if self._is_weight(tensor.name):
                continue

            shape = tensor.shape
            if not shape or len(shape) <= axis:
                continue

            dim_size = shape[axis]
            if dim_size <= 0:
                # 动态维度，无法确定
                continue

            dim_sizes.append(dim_size)

        if not dim_sizes:
            # 没有有效维度（所有输入都是权重），不能分割
            return (False, None, None)

        # 检查初始 parts 是否适用于所有维度
        initial_valid = all(
            dim >= initial_parts and dim % initial_parts == 0
            for dim in dim_sizes
        )

        if initial_valid:
            return (True, initial_parts, None)

        # 计算搜索上限
        max_dim = max(dim_sizes)
        search_limit = min(max_dim, initial_parts * 4, 256)

        # 从 initial_parts + 1 开始向上查找
        for parts in range(initial_parts + 1, search_limit + 1):
            if all(dim >= parts and dim % parts == 0 for dim in dim_sizes):
                return (True, parts, None)

        # 未找到合适的 parts
        dim_info = ", ".join(str(d) for d in dim_sizes)
        warning = (
            f"{op_info.name}: skipped split - dimension(s) [{dim_info}] on axis {axis} "
            f"cannot be evenly split by {initial_parts} (tried up to {search_limit})"
        )
        return (False, None, warning)

    def _get_candidate_axes(
        self,
        op_info: OperatorInfo,
        splitable_axes: SplitableAxes,
        configured_axis: Optional[int],
    ) -> tuple[list[int], bool]:
        """获取候选切分轴列表。

        Returns:
            (candidate_axes, auto_planning_disabled)
        """
        if configured_axis is not None:
            if not splitable_axes.axes:
                return ([], False)
            if configured_axis in splitable_axes.axes:
                return ([configured_axis], False)
            return (sorted(splitable_axes.axes), False)

        (
            matched_rule,
            preferred_axis,
            auto_planning_disabled,
            rule_prefer_axis,
        ) = self._get_axis_rule_preference(op_info.op_type)
        if matched_rule and isinstance(rule_prefer_axis, int) and preferred_axis is not None:
            self._record_integer_axis_rule_match(
                op_info.op_type,
                rule_prefer_axis,
                preferred_axis in splitable_axes.axes,
            )

        if not splitable_axes.axes:
            return ([], auto_planning_disabled)

        fallback_axes = sorted(splitable_axes.axes, key=lambda a: (a != 0, a))

        if auto_planning_disabled:
            return ([], True)

        if matched_rule and preferred_axis is not None and preferred_axis in splitable_axes.axes:
            remaining_axes = [axis for axis in fallback_axes if axis != preferred_axis]
            return ([preferred_axis, *remaining_axes], False)

        return (fallback_axes, False)

    def _get_axis_rule_preference(
        self,
        op_type: str,
    ) -> tuple[bool, Optional[int], bool, object]:
        """获取算子类型的axis_rules偏好。

        Returns:
            (matched_rule, preferred_axis, auto_planning_disabled, rule_prefer_axis)
        """
        for rule in self.config.axis_rules:
            if rule.op_type != op_type:
                continue

            if rule.prefer_axis is None:
                return (True, None, True, rule.prefer_axis)

            if rule.prefer_axis == "batch":
                return (True, 0, False, rule.prefer_axis)

            if isinstance(rule.prefer_axis, bool):
                self._warn_invalid_axis_rule(op_type, rule.prefer_axis)
                return (True, None, False, rule.prefer_axis)

            if isinstance(rule.prefer_axis, int):
                return (True, rule.prefer_axis, False, rule.prefer_axis)

            if isinstance(rule.prefer_axis, str):
                self._warn_invalid_axis_rule(op_type, rule.prefer_axis)
                return (True, None, False, rule.prefer_axis)

            return (True, None, False, rule.prefer_axis)

        return (False, None, False, None)

    def _is_weight(self, tensor_name: str) -> bool:
        """检查张量是否是权重（常数）

        Args:
            tensor_name: 张量名称

        Returns:
            True如果是权重，False否则
        """
        # 检查是否在initializer中
        if any(init.name == tensor_name for init in self.analyzer.model.graph.initializer):
            return True

        # 检查是否由Constant节点产生
        for node in self.analyzer.model.graph.node:
            if node.op_type == "Constant" and tensor_name in node.output:
                return True

        # 找到产生该张量的节点
        producer_node = None
        for node in self.analyzer.model.graph.node:
            if tensor_name in node.output:
                producer_node = node
                break

        if not producer_node:
            return False

        # 检查是否由DequantizeLinear或QuantizeLinear节点产生，且所有输入都是权重
        # 这种情况通常出现在QDQ格式的量化模型中，其中权重的QDQ节点不应被视为数据流
        if producer_node.op_type in ("DequantizeLinear", "QuantizeLinear"):
            # 检查QDQ节点的所有输入是否都是initializers（直接权重）
            all_inputs_are_initializers = True
            for input_name in producer_node.input:
                if not input_name:
                    continue
                if not any(init.name == input_name for init in self.analyzer.model.graph.initializer):
                    all_inputs_are_initializers = False
                    break
            if all_inputs_are_initializers:
                return True

        # 检查是否由Identity节点产生，且输入是权重
        if producer_node.op_type == "Identity":
            for input_name in producer_node.input:
                if input_name and self._is_weight(input_name):
                    return True

        return False

    def _get_operator_config(self, op_name: str) -> tuple[int, Optional[int]]:
        """获取算子的切分配置

        优先级：算子精确匹配 > 通配符匹配 > 全局配置

        Args:
            op_name: 算子名称

        Returns:
            (parts, axis) 元组
        """
        # 1. 精确匹配 (O(1))
        if op_name in self._exact_match_patterns:
            op_config = self._exact_match_patterns[op_name]
            return (op_config.parts, op_config.axis)

        # 2. 通配符匹配 (按配置顺序遍历，但使用预编译的regex)
        for compiled_pattern in self._compiled_patterns:
            if compiled_pattern.regex.match(op_name):
                return (compiled_pattern.config.parts, compiled_pattern.config.axis)

        # 3. 全局配置
        return (self.config.global_config.default_parts, None)

    def _compile_config_patterns(self) -> None:
        """编译配置中的通配符模式

        将精确匹配和通配符模式分离，预编译通配符为正则表达式。
        这样精确匹配是O(1)，通配符匹配只需要遍历通配符模式列表。
        """
        self._exact_match_patterns.clear()
        self._compiled_patterns.clear()

        for pattern, config in self.config.operators.items():
            # 检查是否为通配符模式
            is_wildcard = any(char in pattern for char in "*?[")

            if not is_wildcard:
                # 精确匹配，存入字典实现O(1)查找
                self._exact_match_patterns[pattern] = config
            else:
                # 通配符模式，预编译为正则表达式
                # fnmatch.translate将通配符转换为正则表达式
                regex_str = translate(pattern)
                compiled_regex = compile(regex_str)
                self._compiled_patterns.append(
                    CompiledPattern(
                        pattern=pattern,
                        regex=compiled_regex,
                        is_wildcard=True,
                        config=config,
                    )
                )

    def get_splitable_operators(self) -> list[OperatorInfo]:
        """获取所有可切分的算子列表

        Returns:
            可切分算子列表
        """
        if not self._splitable_ops:
            self._analyze_splitability()

        return [op_info for op_info, splitable in self._splitable_ops.values() if splitable.axes]

    def get_warnings(self) -> list[str]:
        """获取收集的警告信息

        Returns:
            警告信息列表
        """
        return self._warnings.copy()

    def _add_warning(self, message: str) -> None:
        """添加警告信息

        Args:
            message: 警告内容
        """
        self._warnings.append(message)

    def _warn_invalid_axis_rule(self, op_type: str, prefer_axis: object) -> None:
        """为无效axis_rules配置发出一次警告。"""
        warning_key = (op_type, repr(prefer_axis))
        if warning_key in self._invalid_axis_rule_warnings:
            return

        self._invalid_axis_rule_warnings.add(warning_key)
        self._add_warning(
            f"{op_type}: unsupported axis_rules prefer_axis={prefer_axis!r} "
            f"({type(prefer_axis).__name__}); falling back to default axis ordering"
        )

    def _record_integer_axis_rule_match(
        self,
        op_type: str,
        prefer_axis: int,
        usable: bool,
    ) -> None:
        """记录整数axis_rules在当前规划中的匹配和可用性。"""
        rule_key = (op_type, prefer_axis)
        previous_usable = self._matched_integer_axis_rules.get(rule_key, False)
        self._matched_integer_axis_rules[rule_key] = previous_usable or usable

    def _warn_never_usable_integer_axis_rules(self) -> None:
        """为当前规划中从未可用的整数axis_rules发出一次警告。"""
        for (op_type, prefer_axis), was_usable in self._matched_integer_axis_rules.items():
            if was_usable:
                continue
            self._add_warning(
                f"{op_type}: axis_rules prefer_axis={prefer_axis!r} "
                "never usable for matching operators in this planning run; "
                "falling back to default axis ordering"
            )
