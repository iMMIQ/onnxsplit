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
        self._splitable_ops: dict[str, tuple[OperatorInfo, SplitableAxes]] = {}
        self._compiled_patterns: list[CompiledPattern] = []
        self._exact_match_patterns: dict[str, OperatorConfig] = {}
        self._compile_config_patterns()

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

        # 确定候选切分轴（按优先级排序）
        candidate_axes = []
        if axis is not None:
            # 用户指定了轴
            if axis in splitable_axes.axes:
                candidate_axes = [axis]
            else:
                # 指定的轴不可切，尝试其他可切分轴
                candidate_axes = sorted(splitable_axes.axes)
        else:
            # 自动选择轴：优先batch(axis=0)，然后按数字顺序
            candidate_axes = sorted(splitable_axes.axes, key=lambda a: (a != 0, a))

        if not candidate_axes:
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
        # 检查所有输入张量
        for tensor in op_info.input_tensors:
            # 跳过权重（常数）
            if self._is_weight(tensor.name):
                continue

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

        return True

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
            # 没有有效维度，使用初始值
            return (True, initial_parts, None)

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
