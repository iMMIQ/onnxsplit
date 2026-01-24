"""图变换执行器"""
import copy
from typing import Optional

import onnx
from onnx import helper

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.node_clone import clone_node, generate_split_name
from onnxsplit.transform.split_concat import create_split_node, create_concat_node, get_slice_initializers


class GraphTransformer:
    """图变换执行器

    根据切分方案对ONNX图进行变换。
    """

    def __init__(self, analyzer: ModelAnalyzer):
        """初始化变换器

        Args:
            analyzer: 模型分析器
        """
        self.analyzer = analyzer
        self._node_map: dict[str, list[onnx.NodeProto]] = {}
        self._tensor_map: dict[str, str] = {}

    def apply_split_plan(self, plan: SplitPlan) -> onnx.ModelProto:
        """应用切分方案到模型

        Args:
            plan: 切分方案

        Returns:
            变换后的新模型

        Raises:
            ValueError: 算子不存在
        """
        if not plan.is_split:
            # 不需要切分，返回原模型的副本
            return copy.deepcopy(self.analyzer.model)

        # 获取原始算子
        original_op = self.analyzer.get_operator(plan.operator_name)
        if original_op is None:
            raise ValueError(f"Operator not found: {plan.operator_name}")

        # 克隆模型
        new_model = copy.deepcopy(self.analyzer.model)
        new_graph = new_model.graph

        # 找到要切分的节点
        target_node = None
        for node in new_graph.node:
            if node.name == plan.operator_name:
                target_node = node
                break

        if target_node is None:
            raise ValueError(f"Node not found in graph: {plan.operator_name}")

        # 克隆节点
        cloned_nodes = []
        for i in range(plan.parts):
            new_outputs = [f"{out}_{i}" for out in target_node.output]
            cloned = clone_node(
                target_node,
                suffix=f"_split_{i}",
                new_outputs=new_outputs,
            )
            cloned_nodes.append(cloned)

        # 移除原始节点，添加克隆节点
        nodes_to_remove = []
        nodes_to_add = []
        for node in new_graph.node:
            if node.name == plan.operator_name:
                nodes_to_remove.append(node)
                # 插入输入切分（如果需要）
                if self._needs_input_split(target_node):
                    split_nodes = self._create_input_splits(target_node, plan)
                    nodes_to_add.extend(split_nodes)
                # 添加克隆节点
                nodes_to_add.extend(cloned_nodes)
                # 插入输出合并（如果需要）
                if self._needs_output_merge(target_node):
                    concat_nodes = self._create_output_merges(target_node, plan)
                    nodes_to_add.extend(concat_nodes)
                break

        # 更新图
        self._update_graph_nodes(new_graph, nodes_to_remove, nodes_to_add)

        # 运行形状推断
        new_model = onnx.shape_inference.infer_shapes(new_model)

        return new_model

    def _needs_input_split(self, node: onnx.NodeProto) -> bool:
        """检查是否需要在输入端插入Split"""
        for input_name in node.input:
            if not input_name:
                continue
            producer = self.analyzer.get_tensor_producer(input_name)
            if producer is None or producer != node.name:
                return True
        return False

    def _needs_output_merge(self, node: onnx.NodeProto) -> bool:
        """检查是否需要在输出端插入Concat"""
        for output_name in node.output:
            consumers = self.analyzer.get_tensor_consumers(output_name)
            if consumers or self._is_model_output(output_name):
                return True
        return False

    def _is_model_output(self, tensor_name: str) -> bool:
        """检查张量是否是模型输出"""
        return any(output.name == tensor_name for output in self.analyzer.model.graph.output)

    def _create_input_splits(
        self, node: onnx.NodeProto, plan: SplitPlan
    ) -> list[onnx.NodeProto]:
        """创建输入切分节点"""
        split_nodes = []

        for input_name in node.input:
            if not input_name:
                continue

            if self._is_weight(input_name):
                continue

            split_node = create_split_node(
                input_name=input_name,
                axis=plan.axis,
                parts=plan.parts,
                output_prefix=f"{input_name}_split",
                node_name=f"split_{input_name}",
            )
            split_nodes.append(split_node)

        return split_nodes

    def _create_output_merges(
        self, node: onnx.NodeProto, plan: SplitPlan
    ) -> list[onnx.NodeProto]:
        """创建输出合并节点"""
        concat_nodes = []

        for i, output_name in enumerate(node.output):
            split_outputs = [f"{output_name}_{j}" for j in range(plan.parts)]

            concat_node = create_concat_node(
                input_names=split_outputs,
                output_name=output_name,
                axis=plan.axis,
                node_name=f"concat_{output_name}",
            )
            concat_nodes.append(concat_node)

        return concat_nodes

    def _is_weight(self, tensor_name: str) -> bool:
        """检查张量是否是权重"""
        return any(
            init.name == tensor_name
            for init in self.analyzer.model.graph.initializer
        )

    def _update_graph_nodes(
        self,
        graph: onnx.GraphProto,
        to_remove: list[onnx.NodeProto],
        to_add: list[onnx.NodeProto],
    ) -> None:
        """更新图的节点列表"""
        nodes_to_keep = []
        remove_names = {n.name for n in to_remove}
        for node in graph.node:
            if node.name not in remove_names:
                nodes_to_keep.append(node)

        graph.node.clear()
        graph.node.extend(nodes_to_keep)

        for node in to_add:
            graph.node.append(node)
