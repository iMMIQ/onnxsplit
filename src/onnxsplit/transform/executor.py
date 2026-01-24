"""图变换执行器

负责执行图变换操作，包括算子切分、节点克隆、数据流重连等。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import onnx
from onnx import GraphProto, ModelProto, NodeProto

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.node_clone import clone_node
from onnxsplit.transform.reconnect import generate_reconnect_plan
from onnxsplit.transform.split_concat import (
    create_concat_node,
    create_slice_node,
    create_split_node,
    get_slice_initializers,
)


@dataclass
class TransformContext:
    """变换上下文

    存储变换过程中的状态和临时数据。

    Attributes:
        model: 原始ONNX模型
        batch_dim: 批次维度
        original_graph: 原始计算图
        new_nodes: 新添加的节点
        new_inputs: 新添加的输入
        new_outputs: 新添加的输出
        new_initializers: 新添加的初始器
        _analyzer: 缓存的模型分析器
    """

    model: ModelProto
    batch_dim: int = 0
    original_graph: Optional[GraphProto] = None
    new_nodes: list[NodeProto] = field(default_factory=list)
    new_inputs: list = field(default_factory=list)
    new_outputs: list = field(default_factory=list)
    new_initializers: list = field(default_factory=list)
    _analyzer: Optional[ModelAnalyzer] = None

    def __post_init__(self) -> None:
        """初始化后处理"""
        if self.original_graph is None:
            self.original_graph = self.model.graph

    @property
    def analyzer(self) -> ModelAnalyzer:
        """获取模型分析器"""
        if self._analyzer is None:
            self._analyzer = ModelAnalyzer.from_model_proto(self.model)
        return self._analyzer

    def get_operator(self, name: str) -> Optional[NodeProto]:
        """通过名称获取算子节点

        Args:
            name: 算子名称

        Returns:
            算子节点，不存在时返回None
        """
        for node in self.original_graph.node:
            if node.name == name:
                return node
        return None

    def add_node(self, node: NodeProto) -> None:
        """添加新节点

        Args:
            node: 要添加的节点
        """
        self.new_nodes.append(node)

    def add_initializer(self, initializer) -> None:
        """添加初始器

        Args:
            initializer: 要添加的初始器张量
        """
        self.new_initializers.append(initializer)


@dataclass
class TransformResult:
    """变换结果

    Attributes:
        success: 是否成功
        error_message: 错误信息
        split_operators: 被切分的算子列表
        new_nodes: 新生成的节点列表
        metrics: 变换指标
    """

    success: bool = True
    error_message: str = ""
    split_operators: list[str] = field(default_factory=list)
    new_nodes: list[str] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化后处理"""
        if "split_count" not in self.metrics:
            self.metrics["split_count"] = 0


class GraphTransformer:
    """图变换执行器

    负责执行各种图变换操作，包括算子切分、节点克隆、数据流重连等。

    Attributes:
        model: 原始ONNX模型
        batch_dim: 批次维度
        _analyzer: 模型分析器
        _context: 变换上下文
        _split_results: 切分结果记录
    """

    def __init__(self, model: ModelProto, batch_dim: int = 0) -> None:
        """初始化变换器

        Args:
            model: ONNX模型
            batch_dim: 批次维度
        """
        self.model = model
        self.batch_dim = batch_dim
        self._analyzer = ModelAnalyzer.from_model_proto(model)
        self._context: Optional[TransformContext] = None
        self._split_results: dict[str, list[NodeProto]] = {}

    def create_context(self) -> TransformContext:
        """创建变换上下文

        Returns:
            变换上下文
        """
        self._context = TransformContext(self.model, self.batch_dim)
        return self._context

    @property
    def context(self) -> TransformContext:
        """获取或创建变换上下文"""
        if self._context is None:
            self._context = self.create_context()
        return self._context

    def apply_split(self, plan: SplitPlan) -> TransformResult:
        """应用单个切分方案

        Args:
            plan: 切分方案

        Returns:
            变换结果
        """
        return self.apply_splits([plan])

    def apply_splits(self, plans: list[SplitPlan]) -> TransformResult:
        """应用多个切分方案

        Args:
            plans: 切分方案列表

        Returns:
            变换结果
        """
        result = TransformResult()
        ctx = self.context

        if not plans:
            return result

        for plan in plans:
            if not plan.is_split:
                continue

            # 查找目标算子
            node = ctx.get_operator(plan.operator_name)
            if node is None:
                # 算子不存在，跳过
                continue

            # 执行切分
            split_nodes = self._split_operator(node, plan)
            if split_nodes:
                self._split_results[plan.operator_name] = split_nodes
                result.split_operators.append(plan.operator_name)
                result.new_nodes.extend(n.name for n in split_nodes if n.name)

        result.metrics["split_count"] = len(result.split_operators)
        return result

    def _split_operator(self, node: NodeProto, plan: SplitPlan) -> list[NodeProto]:
        """切分单个算子

        Args:
            node: 算子节点
            plan: 切分方案

        Returns:
            切分后的节点列表
        """
        ctx = self.context
        split_nodes = []

        # 获取输入输出形状
        input_name = node.input[0] if node.input else ""
        output_name = node.output[0] if node.output else ""

        # 在输入端添加Split节点
        split_outputs = [f"{plan.operator_name}_split_{i}.{input_name}" for i in range(plan.parts)]
        split_node = create_split_node(
            input_name=input_name,
            axis=plan.axis,
            parts=plan.parts,
            output_prefix=f"{plan.operator_name}_split",
            node_name=f"split_{plan.operator_name}_input",
        )
        ctx.add_node(split_node)

        # 克隆原始节点为多个副本
        for i in range(plan.parts):
            new_output = f"{plan.operator_name}_split_{i}.{output_name}"
            new_node = clone_node(
                node,
                suffix=f"_split_{i}",
                new_outputs=[new_output],
                new_name=f"{node.name}_split_{i}" if node.name else None,
            )

            # 修改输入为split后的输出
            new_node.input[0] = split_outputs[i]

            ctx.add_node(new_node)
            split_nodes.append(new_node)

        # 在输出端添加Concat节点
        concat_inputs = [n.output[0] for n in split_nodes if n.output]
        concat_node = create_concat_node(
            input_names=concat_inputs,
            output_name=output_name,
            axis=plan.axis,
            node_name=f"concat_{plan.operator_name}_output",
        )
        ctx.add_node(concat_node)

        return split_nodes

    def reconnect_dataflow(
        self,
        src_op: str,
        dst_op: str,
        src_parts: int,
        dst_parts: int,
        src_output: str,
        dst_input: str,
        batch_size: int,
    ) -> None:
        """重新连接数据流

        当源算子和目标算子的切分数不同时，需要插入额外的Slice/Concat节点。

        Args:
            src_op: 源算子名称
            dst_op: 目标算子名称
            src_parts: 源算子切分数
            dst_parts: 目标算子切分数
            src_output: 源算子输出名称
            dst_input: 目标算子输入名称
            batch_size: 批次大小
        """
        ctx = self.context

        # 生成重连计划
        plan = generate_reconnect_plan(
            src_op=src_op,
            dst_op=dst_op,
            src_parts=src_parts,
            dst_parts=dst_parts,
            batch_size=batch_size,
            src_output=src_output,
            dst_input=dst_input,
            axis=self.batch_dim,
        )

        # 执行重连操作
        for slice_op in plan.slice_operations:
            slice_node = create_slice_node(
                input_name=slice_op.input_tensor,
                output_name=slice_op.output_tensor,
                starts=[slice_op.start] if slice_op.axis == 0 else [0, slice_op.start],
                ends=[slice_op.end] if slice_op.axis == 0 else [0, slice_op.end],
                axes=[slice_op.axis],
                node_name=f"slice_{src_op}_{dst_op}",
            )
            ctx.add_node(slice_node)

            # 添加Slice节点的初始器
            for init in get_slice_initializers(slice_node):
                ctx.add_initializer(init)

        for concat_op in plan.concat_operations:
            concat_node = create_concat_node(
                input_names=concat_op.input_tensors,
                output_name=concat_op.output_tensor,
                axis=concat_op.axis,
                node_name=f"concat_{src_op}_{dst_op}",
            )
            ctx.add_node(concat_node)

        for split_op in plan.split_operations:
            split_node = create_split_node(
                input_name=split_op.input_tensor,
                axis=split_op.axis,
                parts=len(split_op.output_tensors),
                output_prefix=split_op.output_tensors[0].rsplit("_", 1)[0],
                split_sizes=split_op.split_sizes,
                node_name=f"split_{src_op}_{dst_op}",
            )
            ctx.add_node(split_node)

    def build(self) -> ModelProto:
        """构建变换后的模型

        Returns:
            变换后的ONNX模型
        """
        # 创建新模型
        new_model = onnx.helper.make_model(
            graph=self._build_graph(),
            producer_name=self.model.producer_name,
            producer_version=self.model.producer_version,
        )
        new_model.ir_version = self.model.ir_version

        # 复制opset_import
        if self.model.opset_import:
            for opset in self.model.opset_import:
                new_model.opset_import.append(opset)

        return new_model

    def _build_graph(self) -> GraphProto:
        """构建变换后的计算图

        Returns:
            变换后的计算图
        """
        # 创建新图
        graph = onnx.helper.make_graph(
            nodes=self._collect_all_nodes(),
            name=self.model.graph.name or "split_model",
            inputs=list(self.model.graph.input),
            outputs=list(self.model.graph.output),
            initializer=self._collect_all_initializers(),
        )

        # 复制value_info
        if self.model.graph.value_info:
            for value_info in self.model.graph.value_info:
                graph.value_info.append(value_info)

        return graph

    def _collect_all_nodes(self) -> list[NodeProto]:
        """收集所有节点（包括原始节点和新节点）

        Returns:
            所有节点的列表
        """
        ctx = self.context
        all_nodes = []

        # 添加原始节点（排除被切分的节点）
        split_op_names = set(self._split_results.keys())
        for node in self.model.graph.node:
            if node.name not in split_op_names:
                all_nodes.append(node)

        # 添加新节点
        all_nodes.extend(ctx.new_nodes)

        return all_nodes

    def _collect_all_initializers(self) -> list:
        """收集所有初始器

        Returns:
            所有初始器的列表
        """
        ctx = self.context
        all_initializers = []

        # 添加原始初始器
        if self.model.graph.initializer:
            all_initializers.extend(list(self.model.graph.initializer))

        # 添加新初始器
        all_initializers.extend(ctx.new_initializers)

        return all_initializers

    def execute_and_save(self, plans: list[SplitPlan], output_path: Path | str) -> TransformResult:
        """执行变换并保存模型

        Args:
            plans: 切分方案列表
            output_path: 输出文件路径

        Returns:
            变换结果
        """
        # 应用切分
        result = self.apply_splits(plans)

        # 构建新模型
        new_model = self.build()

        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型
        onnx.save(new_model, str(output_path))

        return result

    def get_metrics(self) -> dict[str, int]:
        """获取变换指标

        Returns:
            指标字典
        """
        return {
            "split_count": len(self._split_results),
            "new_node_count": len(self.context.new_nodes),
            "original_node_count": len(self.model.graph.node),
        }
