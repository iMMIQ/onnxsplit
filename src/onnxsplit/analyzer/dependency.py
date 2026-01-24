"""依赖关系图构建"""

from collections import defaultdict, deque
from dataclasses import dataclass

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.analyzer.operator import OperatorInfo


@dataclass(frozen=True)
class DependencyEdge:
    """依赖边

    Attributes:
        src: 源算子名称
        dst: 目标算子名称
        tensor_name: 传递的张量名称
    """

    src: str
    dst: str
    tensor_name: str


class DependencyNode:
    """依赖图节点

    Attributes:
        name: 算子名称
        op_type: 算子类型
        info: 算子信息
    """

    def __init__(self, name: str, op_type: str, info: OperatorInfo):
        self.name = name
        self.op_type = op_type
        self.info = info

    def __repr__(self) -> str:
        return f"DependencyNode(name={self.name!r}, op_type={self.op_type!r})"


class DependencyGraph:
    """算子依赖关系图

    有向图，表示算子间的数据依赖关系。
    边 A -> B 表示 A 的输出是 B 的输入。
    """

    def __init__(self):
        """初始化空的依赖图"""
        self.nodes: dict[str, DependencyNode] = {}
        self._outgoing: dict[str, list[DependencyEdge]] = defaultdict(list)
        self._incoming: dict[str, list[DependencyEdge]] = defaultdict(list)

    @classmethod
    def build(cls, analyzer: ModelAnalyzer) -> "DependencyGraph":
        """从模型分析器构建依赖图

        Args:
            analyzer: 模型分析器

        Returns:
            DependencyGraph实例
        """
        graph = cls()

        # 添加所有算子作为节点
        for op_info in analyzer.get_operators():
            node = DependencyNode(op_info.name, op_info.op_type, op_info)
            graph.add_node(node)

        # 构建边（基于数据流）
        for op_info in analyzer.get_operators():
            for input_name in op_info.input_names:
                # 找到产生这个输入的算子
                producer = analyzer.get_tensor_producer(input_name)
                if producer and producer in graph.nodes:
                    # 创建边: producer -> current_op
                    edge = DependencyEdge(producer, op_info.name, input_name)
                    graph.add_edge(edge)

        return graph

    def add_node(self, node: DependencyNode) -> None:
        """添加节点

        Args:
            node: 依赖图节点
        """
        self.nodes[node.name] = node

    def add_edge(self, edge: DependencyEdge) -> None:
        """添加边

        Args:
            edge: 依赖边
        """
        self._outgoing[edge.src].append(edge)
        self._incoming[edge.dst].append(edge)

    def get_outgoing_edges(self, node_name: str) -> list[DependencyEdge]:
        """获取节点的出边

        Args:
            node_name: 节点名称

        Returns:
            出边列表
        """
        return self._outgoing.get(node_name, [])

    def get_incoming_edges(self, node_name: str) -> list[DependencyEdge]:
        """获取节点的入边

        Args:
            node_name: 节点名称

        Returns:
            入边列表
        """
        return self._incoming.get(node_name, [])

    def get_predecessors(self, node_name: str) -> set[str]:
        """获取前驱节点集合

        Args:
            node_name: 节点名称

        Returns:
            前驱节点名称集合
        """
        return {edge.src for edge in self._incoming.get(node_name, [])}

    def get_successors(self, node_name: str) -> set[str]:
        """获取后继节点集合

        Args:
            node_name: 节点名称

        Returns:
            后继节点名称集合
        """
        return {edge.dst for edge in self._outgoing.get(node_name, [])}

    def get_source_nodes(self) -> list[DependencyNode]:
        """获取源节点（无入边的节点）

        Returns:
            源节点列表
        """
        return [node for name, node in self.nodes.items() if not self._incoming.get(name)]

    def get_sink_nodes(self) -> list[DependencyNode]:
        """获取汇节点（无出边的节点）

        Returns:
            汇节点列表
        """
        return [node for name, node in self.nodes.items() if not self._outgoing.get(name)]

    def topological_sort(self) -> list[str]:
        """执行拓扑排序

        Returns:
            拓扑排序后的节点名称列表

        Raises:
            ValueError: 图中存在环
        """
        # Kahn算法
        in_degree = {name: 0 for name in self.nodes}
        for edges in self._incoming.values():
            for edge in edges:
                in_degree[edge.dst] += 1

        # 找出所有入度为0的节点
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # 减少后继节点的入度
            for edge in self._outgoing.get(node, []):
                in_degree[edge.dst] -= 1
                if in_degree[edge.dst] == 0:
                    queue.append(edge.dst)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")

        return result

    def has_path(self, src: str, dst: str) -> bool:
        """检查是否存在从src到dst的路径

        Args:
            src: 源节点名称
            dst: 目标节点名称

        Returns:
            存在路径返回True
        """
        if src == dst:
            return True

        visited = set()
        queue = deque([src])

        while queue:
            node = queue.popleft()
            if node == dst:
                return True

            if node in visited:
                continue
            visited.add(node)

            for edge in self._outgoing.get(node, []):
                if edge.dst not in visited:
                    queue.append(edge.dst)

        return False

    def has_cycle(self) -> bool:
        """检测图中是否存在环

        Returns:
            存在环返回True
        """
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def __repr__(self) -> str:
        return (
            f"DependencyGraph(nodes={len(self.nodes)}, "
            f"edges={sum(len(e) for e in self._outgoing.values())})"
        )
