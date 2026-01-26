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

        # 确定切分轴
        if axis is not None:
            # 用户指定了轴，检查是否可切
            if axis not in splitable_axes.axes:
                # 指定的轴不可切，回退到默认或跳过
                if splitable_axes.axes:
                    axis = next(iter(splitable_axes.axes))
                else:
                    return None
        else:
            # 自动选择轴：优先选择batch(axis=0)
            if 0 in splitable_axes.axes:
                axis = 0
            else:
                # 选择第一个可切分轴
                axis = min(splitable_axes.axes) if splitable_axes.axes else None

        if axis is None:
            return None

        # 检查输入形状是否支持均匀分割
        if not self._is_evenly_splittable(op_info, axis, parts):
            return None

        return SplitPlan(
            operator_name=op_info.name,
            parts=parts,
            axis=axis,
            reason=splitable_axes.reason,
        )

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
