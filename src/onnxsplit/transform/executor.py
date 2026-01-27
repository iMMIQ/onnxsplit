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
        # 检查图中是否已存在对同一输入的Split节点，避免重复创建
        input_split_map = {}  # 原始输入名 -> 切分后的输出名列表

        # 检查是否有上游切分可以复用
        has_upstream_split_to_reuse = False
        for input_name in target_node.input:
            if not input_name or self._is_weight(input_name):
                continue
            upstream_info = self._find_upstream_split_info(
                self.analyzer.model.graph, input_name, target_node
            )
            if upstream_info is not None:
                upstream_parts, upstream_axis, _ = upstream_info
                if upstream_parts == plan.parts and upstream_axis == plan.axis:
                    has_upstream_split_to_reuse = True
                    break

        # 只有在没有上游切分可复用时，才检查是否需要创建新的split
        needs_new_split = self._needs_input_split(
            target_node, plan_parts=plan.parts, plan_axis=plan.axis
        )

        if needs_new_split or has_upstream_split_to_reuse:
            split_nodes, input_split_map = self._create_input_splits(
                new_graph, target_node, plan
            )
            nodes_to_add.extend(split_nodes)

        # 如果没有输入被split（由于形状不兼容等原因），返回原始模型
        if not input_split_map and self._needs_input_split(target_node, plan_parts=plan.parts, plan_axis=plan.axis):
            # 需要split但无法split（形状不兼容），返回原模型副本
            return copy.deepcopy(self.analyzer.model)

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

    def _needs_input_split(
        self, node: onnx.NodeProto, plan_parts: int | None = None, plan_axis: int | None = None
    ) -> bool:
        """检查是否需要在输入端插入Split

        Args:
            node: 要检查的节点
            plan_parts: 切分方案的份数（用于检测上游是否已按相同配置切分）
            plan_axis: 切分方案的轴（用于检测上游是否已按相同配置切分）

        Returns:
            True 如果需要插入Split，False 否则
        """
        for input_name in node.input:
            if not input_name:
                continue

            # 检查是否是权重
            if self._is_weight(input_name):
                continue

            producer = self.analyzer.get_tensor_producer(input_name)
            if producer is None or producer != node.name:
                # 如果提供了plan信息，检查上游是否已按相同配置切分
                if plan_parts is not None and plan_axis is not None:
                    upstream_info = self._find_upstream_split_info(
                        self.analyzer.model.graph, input_name, node
                    )
                    if upstream_info is not None:
                        upstream_parts, upstream_axis, _ = upstream_info
                        # 上游已按相同配置切分，不需要再split
                        if upstream_parts == plan_parts and upstream_axis == plan_axis:
                            return False
                return True
        return False

    def _find_existing_split(
        self,
        graph: onnx.GraphProto,
        input_name: str,
        axis: int,
        parts: int,
    ) -> list[str] | None:
        """查找图中是否已存在对指定输入的Split节点

        如果存在一个Split节点，它满足：
        1. 输入是input_name
        2. axis属性匹配
        3. 输出数量等于parts

        则返回该Split节点的输出列表，否则返回None。

        Args:
            graph: ONNX图
            input_name: 要查找的输入张量名称
            axis: 切分轴
            parts: 切分份数

        Returns:
            已存在的Split节点的输出列表，如果不存在则返回None
        """
        for node in graph.node:
            if node.op_type != "Split":
                continue

            # 检查输入是否匹配
            if not node.input or node.input[0] != input_name:
                continue

            # 检查输出数量是否匹配
            if len(node.output) != parts:
                continue

            # 检查axis属性是否匹配
            node_axis = None
            for attr in node.attribute:
                if attr.name == "axis":
                    node_axis = attr.i
                    break

            if node_axis == axis:
                # 找到匹配的Split节点，返回其输出
                return list(node.output)

        return None

    def _find_any_existing_split(
        self,
        graph: onnx.GraphProto,
        input_name: str,
        axis: int,
    ) -> tuple[int, onnx.NodeProto] | None:
        """查找图中是否已存在对指定输入的任意Split节点

        无论parts是多少，只要输入和axis匹配就返回。

        Args:
            graph: ONNX图
            input_name: 要查找的输入张量名称
            axis: 切分轴

        Returns:
            (parts, split_node) 元组，如果不存在则返回None
        """
        for node in graph.node:
            if node.op_type != "Split":
                continue

            # 检查输入是否匹配
            if not node.input or node.input[0] != input_name:
                continue

            # 检查axis属性是否匹配
            node_axis = None
            for attr in node.attribute:
                if attr.name == "axis":
                    node_axis = attr.i
                    break

            if node_axis == axis:
                # 找到匹配的Split节点，返回其parts和节点
                return (len(node.output), node)

        return None

    def _find_any_split_on_input(
        self,
        graph: onnx.GraphProto,
        input_name: str,
    ) -> tuple[int, int, onnx.NodeProto] | None:
        """查找图中是否已存在对指定输入的任意Split节点（无论axis）

        如果存在任何split节点使用该输入，则返回其信息。
        这用于防止在不同axis上创建多个split节点使用同一输入。

        Args:
            graph: ONNX图
            input_name: 要查找的输入张量名称

        Returns:
            (parts, axis, split_node) 元组，如果不存在则返回None
        """
        for node in graph.node:
            if node.op_type != "Split":
                continue

            # 检查输入是否匹配
            if not node.input or node.input[0] != input_name:
                continue

            # 找到使用该输入的Split节点，返回其parts、axis和节点
            parts = len(node.output)
            axis = None
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = attr.i
                    break
            return (parts, axis, node)

        return None

    def _find_upstream_split_info(
        self,
        graph: onnx.GraphProto,
        input_name: str,
        node: onnx.NodeProto,
    ) -> tuple[int, int, list[str]] | None:
        """查找上游节点的切分信息

        如果输入来自一个已被切分的节点，返回切分信息。
        支持两种情况：
        1. 输入直接来自切分节点的部分输出（带_split_后缀）
        2. 输入来自Concat节点，而Concat的输入来自切分节点的部分输出

        Args:
            graph: ONNX图
            input_name: 当前节点的输入名称
            node: 当前节点（用于避免自我引用）

        Returns:
            (parts, axis, split_output_names) 元组，如果上游未切分则返回None
            split_output_names 是切分后的输出名称列表
        """
        # Case 2: 输入来自Concat，而Concat的输入来自切分节点
        # 这是主要情况 - 当上游节点被切分后，其输出会通过Concat合并
        producer = self.analyzer.get_tensor_producer(input_name)
        if producer:
            # 找到生产者节点
            producer_node = None
            for graph_node in graph.node:
                if graph_node.name == producer:
                    producer_node = graph_node
                    break

            if producer_node and producer_node.op_type == "Concat":
                # Concat的输入就是切分后的输出
                concat_inputs = list(producer_node.input)
                if not concat_inputs:
                    return None

                # 验证这些输入是否来自同一个源节点的切分
                # 提取基础名称（去掉后缀数字，如 matmul_out_0 -> matmul_out）
                base_names = set()
                for inp in concat_inputs:
                    # 移除_数字后缀（例如 matmul_out_0 -> matmul_out）
                    parts = inp.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base_names.add(parts[0])
                    else:
                        base_names.add(inp)

                # 如果所有输入来自同一个基础名称，说明是切分节点
                if len(base_names) == 1:
                    # 找到对应的split节点获取axis
                    for graph_node in graph.node:
                        if graph_node.op_type == "Split" and set(graph_node.output) == set(concat_inputs):
                            axis = 0
                            for attr in graph_node.attribute:
                                if attr.name == "axis":
                                    axis = attr.i
                                    break
                            return (len(concat_inputs), axis, concat_inputs)

                    # 如果找不到split节点，从concat推断axis
                    axis = 0
                    for attr in producer_node.attribute:
                        if attr.name == "axis":
                            axis = attr.i
                            break
                    return (len(concat_inputs), axis, concat_inputs)

        # Case 1: 输入直接来自切分节点（带_split_后缀）
        # 检查是否有同名的多个_split_输出
        split_outputs = []
        for graph_node in graph.node:
            if graph_node == node:
                continue
            for output_name in graph_node.output:
                # 检查是否是带_split_后缀的同源输出
                # 例如: matmul_out_split_0, matmul_out_split_1 都属于 matmul_out
                if output_name.startswith(input_name + "_split_"):
                    split_outputs.append(output_name)

        if split_outputs:
            # 找到对应的split节点来获取parts和axis
            for graph_node in graph.node:
                if graph_node.op_type == "Split" and len(split_outputs) > 0:
                    # 检查这个split的输出是否匹配我们找到的split_outputs
                    if set(graph_node.output) == set(split_outputs):
                        # 获取axis
                        axis = 0
                        for attr in graph_node.attribute:
                            if attr.name == "axis":
                                axis = attr.i
                                break
                        return (len(split_outputs), axis, split_outputs)

        return None

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

    def _get_tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        """获取张量形状（包括权重）

        Args:
            tensor_name: 张量名称

        Returns:
            张量形状，未知时返回空元组
        """
        # 首先从analyzer的shape_map获取
        shape = self.analyzer._get_tensor_shape(tensor_name)
        if shape:
            return shape

        # 检查initializer中的权重
        for init in self.analyzer.model.graph.initializer:
            if init.name == tensor_name:
                return tuple(init.dims)

        # 检查Constant节点的输出
        for node in self.analyzer.model.graph.node:
            if node.op_type == "Constant" and tensor_name in node.output:
                # 获取Constant的值
                for attr in node.attribute:
                    if attr.name == "value":
                        # 从tensor中获取形状
                        if hasattr(attr, "t") and attr.t:
                            return tuple(attr.t.dims) if attr.t.dims else ()
        return ()

    def _check_weight_shape_compatibility(
        self,
        node: onnx.NodeProto,
        axis: int,
    ) -> bool:
        """检查权重形状是否与split兼容

        如果权重在切分轴上的维度与数据输入相同且大于1，
        则split会导致形状不匹配。

        Args:
            node: 要切分的节点
            axis: 切分轴

        Returns:
            True 如果兼容（可以split），False 如果不兼容
        """
        # 收集所有非权重输入在切分轴上的维度
        data_input_dims = []
        weight_input_dims = []
        has_unknown_shape = False  # 跟踪是否有未知形状

        for input_name in node.input:
            if not input_name:
                continue

            shape = self._get_tensor_shape(input_name)
            if not shape or len(shape) <= axis:
                # 形状未知或在切分轴之外，跳过
                if not shape:
                    has_unknown_shape = True
                continue

            dim = shape[axis]
            if self._is_weight(input_name):
                weight_input_dims.append(dim)
            else:
                data_input_dims.append(dim)

        # 如果有未知形状，假设兼容（保守处理）
        if has_unknown_shape:
            return True

        # 如果没有权重输入，可以split
        if not weight_input_dims:
            return True

        # 如果没有非权重输入，不split
        if not data_input_dims:
            return False

        # 检查是否存在形状冲突
        # 如果权重在切分轴上的维度 > 1 且与数据输入的维度相同，则不兼容
        for weight_dim in weight_input_dims:
            for data_dim in data_input_dims:
                if weight_dim > 1 and weight_dim == data_dim:
                    # 形状冲突：权重的batch维度与数据相同
                    # split数据会导致形状不匹配
                    return False

        return True

    def _create_input_splits(
        self,
        graph: onnx.GraphProto,
        node: onnx.NodeProto,
        plan: SplitPlan,
    ) -> tuple[list[onnx.NodeProto], dict[str, list[str]]]:
        """创建输入切分节点

        检查图中是否已存在对指定输入的Split节点，如果存在则复用，
        避免创建重复的Split节点导致SSA违规。
        同时检测上游节点是否已被切分，如果是则复用上游切分输出。

        Args:
            graph: ONNX图
            node: 要切分的节点
            plan: 切分方案

        Returns:
            (要添加的Split节点列表, 输入名->切分输出名的映射字典)
        """
        # 检查权重形状兼容性
        if not self._check_weight_shape_compatibility(node, plan.axis):
            # 权重形状不兼容，不创建任何split
            return [], {}

        split_nodes = []
        input_split_map = {}

        for input_name in node.input:
            if not input_name:
                continue

            if self._is_weight(input_name):
                continue

            # 检查张量形状是否支持在指定轴上切分
            tensor_shape = self._get_tensor_shape(input_name)
            # 如果形状已知且rank小于等于切分轴，无法切分，跳过
            # 注意：空元组()可能表示标量或未知形状，需要额外检查
            # 只有当非空形状的rank<=axis时才跳过
            if tensor_shape is not None and len(tensor_shape) > 0 and len(tensor_shape) <= plan.axis:
                # 张量的rank小于等于切分轴，无法切分
                continue
            # 对于空元组（可能是标量或未知），如果是标量则跳过
            # 标量在任何axis上都无法切分
            if tensor_shape is not None and len(tensor_shape) == 0 and plan.axis >= 0:
                # 检查是否是真正的标量（在value_info中有shape=[]的信息）
                # 如果是未知形状，我们让运行时决定
                is_scalar = False
                for vi in self.analyzer.model.graph.value_info:
                    if vi.name == input_name:
                        if vi.type.tensor_type.HasField('shape'):
                            if len(vi.type.tensor_type.shape.dim) == 0:
                                is_scalar = True
                        break
                if is_scalar:
                    continue

            # NEW: 首先检查上游是否已按相同配置切分
            upstream_info = self._find_upstream_split_info(graph, input_name, node)
            if upstream_info is not None:
                upstream_parts, upstream_axis, split_outputs = upstream_info
                if upstream_parts == plan.parts and upstream_axis == plan.axis:
                    # 上游已按相同配置切分，直接复用其输出
                    input_split_map[input_name] = split_outputs
                    continue

            # 检查是否存在任何使用该输入的split节点（无论axis）
            any_split = self._find_any_split_on_input(graph, input_name)

            if any_split is not None:
                existing_parts, existing_axis, existing_node = any_split
                if existing_axis == plan.axis:
                    # axis匹配，检查parts
                    if existing_parts == plan.parts:
                        # parts也匹配，复用已存在的Split节点的输出
                        input_split_map[input_name] = list(existing_node.output)
                    else:
                        # axis匹配但parts不匹配，无法复用
                        # 拒绝split以避免创建多个split节点使用同一输入
                        return [], {}
                else:
                    # axis不匹配，拒绝split以避免在不同axis上创建多个split节点
                    # 这会导致节点名称冲突和形状不匹配
                    return [], {}
            else:
                # 没有已存在的Split节点，创建新的
                split_node = create_split_node(
                    input_name=input_name,
                    axis=plan.axis,
                    parts=plan.parts,
                    output_prefix=f"{input_name}_split",
                )
                split_nodes.append(split_node)
                # 使用新创建的Split节点的实际输出
                input_split_map[input_name] = list(split_node.output)

        return split_nodes, input_split_map

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
        """检查张量是否是权重（包括Constant节点和常数计算产生的）"""
        # 检查是否在initializer中
        if any(init.name == tensor_name for init in self.analyzer.model.graph.initializer):
            return True
        # 检查是否由Constant节点产生
        for node in self.analyzer.model.graph.node:
            if node.op_type == "Constant" and tensor_name in node.output:
                return True
        # 检查是否由常数计算节点产生（所有输入都是权重）
        if self._is_constant_computation(tensor_name):
            return True
        return False

    def _is_constant_computation(self, tensor_name: str) -> bool:
        """检查张量是否是常数计算的结果

        如果产生该张量的节点的所有输入都是权重/常数，
        则该节点产生常数输出，应被视为权重。

        Args:
            tensor_name: 张量名称

        Returns:
            True 如果是常数计算的结果，False 否则
        """
        # 找到产生该张量的节点
        producer_node = None
        for node in self.analyzer.model.graph.node:
            if tensor_name in node.output:
                producer_node = node
                break

        # 没有生产者（可能是图输入）
        if producer_node is None:
            return False

        # Constant节点已经在_is_weight中处理
        if producer_node.op_type == "Constant":
            return True

        # 检查所有输入是否都是权重
        for input_name in producer_node.input:
            if not input_name:
                continue
            # 递归检查输入是否为权重
            # 注意：这里不使用_is_weight，避免无限递归
            if not self._is_direct_weight(input_name):
                return False

        # 所有输入都是权重，所以这个节点产生常数输出
        return True

    def _is_direct_weight(self, tensor_name: str) -> bool:
        """检查张量是否是直接的权重（不包括常数计算）

        这是_is_weight的非递归版本，用于避免无限递归。

        Args:
            tensor_name: 张量名称

        Returns:
            True 如果是直接权重，False 否则
        """
        # 检查是否在initializer中
        if any(init.name == tensor_name for init in self.analyzer.model.graph.initializer):
            return True
        # 检查是否由Constant节点产生
        for node in self.analyzer.model.graph.node:
            if node.op_type == "Constant" and tensor_name in node.output:
                return True
        return False

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
