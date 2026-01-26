"""图变换执行器"""

import copy

import onnx

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.node_clone import clone_node
from onnxsplit.transform.split_concat import create_concat_node, create_split_node


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

        # 找到要切分的节点（通过输出名称匹配，处理无名称节点）
        target_node = None
        for node in new_graph.node:
            # 优先通过节点名称匹配
            if node.name and node.name == plan.operator_name:
                target_node = node
                break
            # 如果节点无名称，尝试通过输出名称生成匹配
            if not node.name and original_op.output_names:
                synthetic_name = f"{node.op_type}_{node.output[0]}"
                if synthetic_name == plan.operator_name:
                    target_node = node
                    break

        if target_node is None:
            raise ValueError(f"Node not found in graph: {plan.operator_name}")

        # 准备节点添加/移除列表
        nodes_to_remove = [target_node]
        nodes_to_add = []

        # 插入输入切分（如果需要），并建立输入到split输出的映射
        input_split_map = {}  # 原始输入名 -> 切分后的输出名列表
        if self._needs_input_split(target_node):
            split_nodes = self._create_input_splits(target_node, plan)
            nodes_to_add.extend(split_nodes)

            # 建立输入映射: 原始输入名 -> [split_0, split_1, ...]
            for input_name in target_node.input:
                if not input_name or self._is_weight(input_name):
                    continue
                split_outputs = [f"{input_name}_split_{i}" for i in range(plan.parts)]
                input_split_map[input_name] = split_outputs

        # 克隆节点，使用切分后的输入
        cloned_nodes = []
        for i in range(plan.parts):
            new_outputs = [f"{out}_{i}" for out in target_node.output]

            # 构建新的输入列表
            new_inputs = []
            for input_name in target_node.input:
                if not input_name:
                    new_inputs.append("")
                elif input_name in input_split_map:
                    # 使用对应的split输出
                    new_inputs.append(input_split_map[input_name][i])
                else:
                    # 权重或不需要切分的输入，保持原样
                    new_inputs.append(input_name)

            cloned = clone_node(
                target_node,
                suffix=f"_split_{i}",
                new_outputs=new_outputs,
                new_inputs=new_inputs,
            )
            cloned_nodes.append(cloned)

        # 添加克隆节点
        nodes_to_add.extend(cloned_nodes)

        # 插入输出合并（如果需要）
        if self._needs_output_merge(target_node):
            concat_nodes = self._create_output_merges(target_node, plan)
            nodes_to_add.extend(concat_nodes)

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

    def _create_input_splits(self, node: onnx.NodeProto, plan: SplitPlan) -> list[onnx.NodeProto]:
        """创建输入切分节点"""
        split_nodes = []

        for input_name in node.input:
            if not input_name:
                continue

            if self._is_weight(input_name):
                continue

            # 不传递node_name，让create_split_node自动生成并清理名称
            split_node = create_split_node(
                input_name=input_name,
                axis=plan.axis,
                parts=plan.parts,
                output_prefix=f"{input_name}_split",
            )
            split_nodes.append(split_node)

        return split_nodes

    def _create_output_merges(self, node: onnx.NodeProto, plan: SplitPlan) -> list[onnx.NodeProto]:
        """创建输出合并节点"""
        concat_nodes = []

        for i, output_name in enumerate(node.output):
            split_outputs = [f"{output_name}_{j}" for j in range(plan.parts)]

            # 不传递node_name，让create_concat_node自动生成并清理名称
            concat_node = create_concat_node(
                input_names=split_outputs,
                output_name=output_name,
                axis=plan.axis,
            )
            concat_nodes.append(concat_node)

        return concat_nodes

    def _is_weight(self, tensor_name: str) -> bool:
        """检查张量是否是权重"""
        return any(init.name == tensor_name for init in self.analyzer.model.graph.initializer)

    def _update_graph_nodes(
        self,
        graph: onnx.GraphProto,
        to_remove: list[onnx.NodeProto],
        to_add: list[onnx.NodeProto],
    ) -> None:
        """更新图的节点列表，保持拓扑顺序

        将to_remove节点从图中移除，并在相同位置插入to_add节点。
        """
        remove_names = {n.name for n in to_remove}

        # 构建新的节点列表，在移除节点的位置插入新节点
        new_nodes = []
        for node in graph.node:
            if node.name in remove_names:
                # 在移除节点的位置插入新节点
                new_nodes.extend(to_add)
            else:
                new_nodes.append(node)

        # 如果所有要移除的节点都不在原节点列表中（防御性代码）
        if not any(n.name in remove_names for n in graph.node):
            # 追加新节点到末尾
            new_nodes.extend(to_add)

        graph.node.clear()
        graph.node.extend(new_nodes)
