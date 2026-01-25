"""Performance benchmark tests"""

import pytest
from pathlib import Path

from onnxsplit.analyzer import ModelAnalyzer
from onnxsplit.config import GlobalConfig, OperatorConfig, SplitConfig
from onnxsplit.splitter import SplitPlanner
from onnxsplit.memory import MemoryEstimator


def benchmark_model_analyzer_get_operators(benchmark):
    """Benchmark: get all operators"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    result = benchmark(analyzer.get_operators)
    assert len(result) >= 1


def benchmark_model_analyzer_get_operator(benchmark):
    """Benchmark: get operator by name (optimized O(1))"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    benchmark(lambda: analyzer.get_operator("conv_0"))


def benchmark_splitter_planner_generate(benchmark):
    """Benchmark: generate split plan"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(global_config=GlobalConfig(default_parts=2))

    planner = SplitPlanner(analyzer, config)

    result = benchmark(planner.generate)
    assert result.original_operators >= 1


def benchmark_splitter_planner_config_lookup(benchmark):
    """Benchmark: config lookup (optimized with compiled patterns)"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators={
            "conv_*": OperatorConfig(parts=4, axis=0),
            "*_output": OperatorConfig(parts=2),
            "conv_0": OperatorConfig(parts=8, axis=0),
        },
    )

    planner = SplitPlanner(analyzer, config)

    benchmark(planner.generate)


def benchmark_memory_estimator_get_peak(benchmark):
    """Benchmark: get peak memory (optimized O(1))"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)
    estimator = MemoryEstimator(analyzer)

    result = benchmark(estimator.get_peak_memory)
    assert result > 0


def benchmark_memory_estimator_build(benchmark):
    """Benchmark: build memory estimator"""
    model_path = Path("tests/fixtures/models/simple_conv.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    benchmark(lambda: MemoryEstimator(analyzer))


@pytest.mark.parametrize("config_patterns", [1, 5, 10, 20])
def benchmark_config_matching_scaling(benchmark, config_patterns):
    """Benchmark: config matching scaling with pattern count"""
    model_path = Path("tests/fixtures/models/model_with_branches.onnx")
    analyzer = ModelAnalyzer.from_path(model_path)

    # Create different numbers of config patterns
    operators = {}
    for i in range(config_patterns):
        operators[f"pattern_{i}_*"] = OperatorConfig(parts=i + 2, axis=0)

    config = SplitConfig(
        global_config=GlobalConfig(default_parts=1),
        operators=operators,
    )

    planner = SplitPlanner(analyzer, config)
    benchmark(planner.generate)
