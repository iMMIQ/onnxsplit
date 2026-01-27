"""测试不同parts导致的悬空split节点bug复现

Bug场景:
- Add算子被切分成3份
- Add的output被Cast和Clip使用
- Cast被切分成2份
- Clip被切分成3份

由于parts不同（2 vs 3），_find_existing_split不会复用已存在的split节点，
导致创建多个split节点使用同一输入。
"""

import onnx
from onnx import TensorProto, helper
import numpy as np

from onnxsplit.analyzer.model import ModelAnalyzer
from onnxsplit.splitter.plan import SplitPlan
from onnxsplit.transform.executor import GraphTransformer


def create_model_with_cast_clip_different_parts() -> onnx.ModelProto:
    """创建一个模型：Add被3个节点使用，其中2个被split，parts不同"""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [6, 4])

    # Add常量
    add_const = helper.make_tensor("add_const", TensorProto.FLOAT, [], [1.0])
    add_const_node = helper.make_node("Constant", [], ["add_bias"], value=add_const)

    # Add节点
    add_node = helper.make_node(
        "Add",
        inputs=["input", "add_bias"],
        outputs=["add_output"],
        name="add_0",
    )

    # Cast节点
    cast_node = helper.make_node(
        "Cast",
        inputs=["add_output"],
        outputs=["cast_output"],
        name="cast_0",
        to=TensorProto.FLOAT,
    )

    # Clip常量
    clip_min = helper.make_tensor("clip_min", TensorProto.FLOAT, [], [0.0])
    clip_min_node = helper.make_node("Constant", [], ["clip_min_value"], value=clip_min)

    clip_max = helper.make_tensor("clip_max", TensorProto.FLOAT, [], [6.0])
    clip_max_node = helper.make_node("Constant", [], ["clip_max_value"], value=clip_max)

    # Clip节点
    clip_node = helper.make_node(
        "Clip",
        inputs=["add_output", "clip_min_value", "clip_max_value"],
        outputs=["clip_output"],
        name="clip_0",
    )

    # 输出张量
    output1 = helper.make_tensor_value_info("cast_output", TensorProto.FLOAT, [6, 4])
    output2 = helper.make_tensor_value_info("clip_output", TensorProto.FLOAT, [6, 4])

    graph = helper.make_graph(
        [add_const_node, add_node, cast_node, clip_min_node, clip_max_node, clip_node],
        "test_model",
        [input_tensor],
        [output1, output2],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def test_different_parts_cause_multiple_splits():
    """测试不同parts导致创建多个split节点"""
    model = create_model_with_cast_clip_different_parts()

    print("\n=== Before any split ===")
    for node in model.graph.node:
        if node.op_type in ["Add", "Cast", "Clip"]:
            print(f"  {node.op_type} {node.name}: inputs={list(node.input)}, outputs={list(node.output)}")

    # 第一阶段：split add_0 (3份)
    analyzer = ModelAnalyzer(model)
    transformer = GraphTransformer(analyzer)
    plan_add = SplitPlan(operator_name="add_0", parts=3, axis=0)
    model_after_add = transformer.apply_split_plan(plan_add)

    print("\n=== After add_0 split (parts=3) ===")
    for node in model_after_add.graph.node:
        if node.op_type in ["Add", "Split", "Concat"]:
            print(f"  {node.op_type} {node.name}: inputs={list(node.input)}, outputs={list(node.output)}")

    # 第二阶段：split cast_0 (2份)
    analyzer2 = ModelAnalyzer(model_after_add)
    transformer2 = GraphTransformer(analyzer2)
    plan_cast = SplitPlan(operator_name="cast_0", parts=2, axis=0)
    model_after_cast = transformer2.apply_split_plan(plan_cast)

    print("\n=== After cast_0 split (parts=2) ===")
    for node in model_after_cast.graph.node:
        if node.op_type in ["Add", "Cast", "Split", "Concat"]:
            print(f"  {node.op_type} {node.name}: inputs={list(node.input)}, outputs={list(node.output)}")

    # 统计针对add_output的split节点
    add_output_splits = []
    for node in model_after_cast.graph.node:
        if node.op_type == "Split" and node.input and node.input[0] == "add_output":
            add_output_splits.append(node)
            print(f"  Split for add_output: {node.name}, outputs={list(node.output)}")

    # 第三阶段：split clip_0 (3份)
    analyzer3 = ModelAnalyzer(model_after_cast)
    transformer3 = GraphTransformer(analyzer3)
    plan_clip = SplitPlan(operator_name="clip_0", parts=3, axis=0)
    model_after_clip = transformer3.apply_split_plan(plan_clip)

    print("\n=== After clip_0 split (parts=3) ===")
    for node in model_after_clip.graph.node:
        if node.op_type in ["Add", "Cast", "Clip", "Split", "Concat"]:
            print(f"  {node.op_type} {node.name}: inputs={list(node.input)}, outputs={list(node.output)}")

    # 统计针对add_output的split节点
    add_output_splits = []
    for node in model_after_clip.graph.node:
        if node.op_type == "Split" and node.input and node.input[0] == "add_output":
            add_output_splits.append(node)

    print(f"\nTotal splits for add_output: {len(add_output_splits)}")
    for node in add_output_splits:
        print(f"  {node.name}: outputs={list(node.output)}, parts={len(node.output)}")

    # BUG：可能创建了两个针对add_output的split节点（2份和3份）
    if len(add_output_splits) > 1:
        print("\nBUG DETECTED: Multiple split nodes for add_output with different parts!")
        # 检查每个split的输出是否被使用
        for node in add_output_splits:
            print(f"\n  Checking {node.name} (outputs={list(node.output)}):")
            for out in node.output:
                used = False
                consumers = []
                for n in model_after_clip.graph.node:
                    if out in n.input:
                        used = True
                        consumers.append(n.name)
                if used:
                    print(f"    {out} -> used by {consumers}")
                else:
                    is_model_output = any(o.name == out for o in model_after_clip.graph.output)
                    if not is_model_output:
                        print(f"    {out} -> DANGLING!")
                    else:
                        print(f"    {out} -> model output")

    # 最终检查：悬空输出
    all_inputs = set()
    all_outputs = set()
    for node in model_after_clip.graph.node:
        for inp in node.input:
            if inp:
                all_inputs.add(inp)
        for out in node.output:
            if out:
                all_outputs.add(out)

    model_outputs = {o.name for o in model_after_clip.graph.output}
    dangling = all_outputs - all_inputs - model_outputs

    print(f"\nFinal dangling outputs: {dangling}")

    # 验证：如果有针对同一输入的多个split节点，这是bug
    input_to_splits = {}
    for node in model_after_clip.graph.node:
        if node.op_type == "Split" and node.input:
            input_name = node.input[0]
            if input_name not in input_to_splits:
                input_to_splits[input_name] = []
            input_to_splits[input_name].append((len(node.output), node.name))

    print("\nSplits by input:")
    for input_name, splits in input_to_splits.items():
        if len(splits) > 1:
            print(f"  {input_name}: {splits} - Multiple splits detected!")
        else:
            parts, name = splits[0]
            print(f"  {input_name}: {parts} parts ({name})")


if __name__ == "__main__":
    test_different_parts_cause_multiple_splits()
