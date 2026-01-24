"""测试依赖关系图构建"""

from pathlib import Path

from onnxsplit.analyzer.dependency import DependencyEdge, DependencyGraph
from onnxsplit.analyzer.model import ModelAnalyzer


def test_dependency_graph_creation():
    """测试创建依赖关系图"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    assert dep_graph is not None
    assert len(dep_graph.nodes) > 0


def test_dependency_graph_nodes():
    """测试依赖图节点"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    # 检查算子节点存在
    node_names = list(dep_graph.nodes.keys())
    assert "conv_0" in node_names


def test_dependency_graph_edges():
    """测试依赖图边"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    # conv_0 -> relu_0
    edges = dep_graph.get_outgoing_edges("conv_0")
    assert len(edges) > 0
    assert any(e.dst == "relu_0" for e in edges)


def test_dependency_graph_incoming_edges():
    """测试获取入边"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    edges = dep_graph.get_incoming_edges("relu_0")
    assert len(edges) > 0
    assert any(e.src == "conv_0" for e in edges)


def test_dependency_graph_topological_order():
    """测试拓扑排序"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    order = dep_graph.topological_sort()
    assert len(order) > 0
    # conv_0 应该在 relu_0 之前
    if "conv_0" in order and "relu_0" in order:
        assert order.index("conv_0") < order.index("relu_0")


def test_dependency_graph_branch_model():
    """测试分支模型的依赖图"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    # 应该有两个Conv分支
    conv_nodes = [n for n in dep_graph.nodes.values() if n.op_type == "Conv"]
    assert len(conv_nodes) == 2

    # 两个Conv都应该连接到Add
    add_node = next((n for n in dep_graph.nodes.values() if n.op_type == "Add"), None)
    assert add_node is not None

    incoming = dep_graph.get_incoming_edges(add_node.name)
    assert len(incoming) == 2


def test_dependency_graph_get_predecessors():
    """测试获取前驱节点"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    predecessors = dep_graph.get_predecessors("relu_0")
    assert "conv_0" in predecessors


def test_dependency_graph_get_successors():
    """测试获取后继节点"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    successors = dep_graph.get_successors("conv_0")
    assert "relu_0" in successors


def test_dependency_graph_source_nodes():
    """测试获取源节点（无入边的节点）"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    sources = dep_graph.get_source_nodes()
    # Conv节点应该是源节点（输入是图输入，不是其他算子的输出）
    conv_nodes = [n for n in sources if n.op_type == "Conv"]
    assert len(conv_nodes) == 2


def test_dependency_graph_sink_nodes():
    """测试获取汇节点（无出边的节点）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    sinks = dep_graph.get_sink_nodes()
    # 最后的Relu应该是汇节点
    relu_nodes = [n for n in sinks if n.op_type == "Relu"]
    assert len(relu_nodes) == 1


def test_dependency_edge_repr():
    """测试依赖边的字符串表示"""
    edge = DependencyEdge(src="A", dst="B", tensor_name="data")
    repr_str = repr(edge)
    assert "A" in repr_str
    assert "B" in repr_str


def test_dependency_graph_has_path():
    """测试路径检查"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    assert dep_graph.has_path("conv_0", "relu_0")
    assert not dep_graph.has_path("relu_0", "conv_0")


def test_dependency_graph_cycles():
    """测试循环检测（简单模型应该无环）"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    dep_graph = DependencyGraph.build(analyzer)

    assert not dep_graph.has_cycle()
