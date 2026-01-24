"""切分规划器"""

import fnmatch
from typing import Optional

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo
from onnxsplit.config import SplitConfig
from onnxsplit.splitter.axis_rules import AxisAnalyzer, SplitableAxes
from onnxsplit.splitter.plan import SplitPlan, SplitReport


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

        return SplitPlan(
            operator_name=op_info.name,
            parts=parts,
            axis=axis,
            reason=splitable_axes.reason,
        )

    def _get_operator_config(self, op_name: str) -> tuple[int, Optional[int]]:
        """获取算子的切分配置

        优先级：算子精确匹配 > 通配符匹配 > 全局配置

        Args:
            op_name: 算子名称

        Returns:
            (parts, axis) 元组
        """
        # 1. 精确匹配
        if op_name in self.config.operators:
            op_config = self.config.operators[op_name]
            return (op_config.parts, op_config.axis)

        # 2. 通配符匹配
        for pattern, op_config in self.config.operators.items():
            if fnmatch.fnmatch(op_name, pattern):
                return (op_config.parts, op_config.axis)

        # 3. 全局配置
        return (self.config.global_config.default_parts, None)

    def get_splitable_operators(self) -> list[OperatorInfo]:
        """获取所有可切分的算子列表

        Returns:
            可切分算子列表
        """
        if not self._splitable_ops:
            self._analyze_splitability()

        return [op_info for op_info, splitable in self._splitable_ops.values() if splitable.axes]
