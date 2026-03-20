# CLI Config And Rule Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose config loading from the CLI and make `axis_rules` plus `overflow_strategy` affect real split behavior.

**Architecture:** Reuse the existing config loader and runner wiring, extend `SplitPlanner` with a small axis-preference layer, and extend `AutoSplitAdjuster` with explicit search strategies. Keep default behavior unchanged when no new config is provided.

**Tech Stack:** Python 3.13, Typer, ONNX, pytest

---

## File Structure

- `src/onnxsplit/cli/parser.py`
  Add the `--config` CLI option and pass it into `RunContext`.
- `src/onnxsplit/cli/runner.py`
  Pass `overflow_strategy` through to the adjuster call path.
- `src/onnxsplit/splitter/planner.py`
  Implement axis rule lookup and candidate-axis ordering without changing splitability analysis.
- `src/onnxsplit/memory/auto_adjust.py`
  Add explicit search strategy support while preserving current defaults.
- `tests/test_cli_parser.py`
  Cover CLI option parsing and help output.
- `tests/test_cli_runner.py`
  Cover end-to-end CLI use of config files.
- `tests/test_splitter_planner.py`
  Cover axis rule precedence and `prefer_axis: null`.
- `tests/test_auto_adjust.py`
  Cover binary vs linear overflow strategies.
- `README.md`
  Align examples and requirements with the real interface.

### Task 1: Expose `--config` From The CLI

**Files:**
- Modify: `src/onnxsplit/cli/parser.py`
- Test: `tests/test_cli_parser.py`
- Test: `tests/test_cli_runner.py`

- [ ] **Step 1: Write the failing parser test for `--config` help and option handling**

Add tests in `tests/test_cli_parser.py` that:

```python
def test_split_command_help_shows_config():
    result = runner.invoke(app, ["split", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout
```

and:

```python
def test_split_command_with_config_option():
    with runner.isolated_filesystem():
        model_path = Path("model.onnx")
        _create_minimal_onnx_model(model_path)
        config_path = Path("config.yaml")
        config_path.write_text("global:\n  default_parts: 1\n")

        result = runner.invoke(
            app,
            ["split", "model.onnx", "--config", "config.yaml", "--no-simplify"],
        )
        assert result.exit_code == 0
```

- [ ] **Step 2: Run tests to verify they fail for the right reason**

Run: `./.venv/bin/pytest tests/test_cli_parser.py::test_split_command_help_shows_config tests/test_cli_parser.py::test_split_command_with_config_option -v`
Expected: FAIL because `--config` is not yet accepted or shown.

- [ ] **Step 3: Write the minimal CLI implementation**

Update `src/onnxsplit/cli/parser.py` so `split()` accepts:

```python
config: Optional[str] = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to YAML configuration file.",
)
```

and passes:

```python
config_path=config,
```

into `RunContext`.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `./.venv/bin/pytest tests/test_cli_parser.py::test_split_command_help_shows_config tests/test_cli_parser.py::test_split_command_with_config_option -v`
Expected: PASS

- [ ] **Step 5: Add runner coverage for config flowing through CLI**

Add or extend a test in `tests/test_cli_runner.py` to confirm a config file used through the command path affects the resulting split report.

- [ ] **Step 6: Run the related CLI tests**

Run: `./.venv/bin/pytest tests/test_cli_parser.py tests/test_cli_runner.py -v`
Expected: PASS

### Task 2: Activate `axis_rules` In Split Planning

**Files:**
- Modify: `src/onnxsplit/splitter/planner.py`
- Test: `tests/test_splitter_planner.py`

- [ ] **Step 1: Write a failing test for type-level axis preference**

Add a test in `tests/test_splitter_planner.py` that builds an operator with multiple splitable axes, configures:

```python
SplitConfig(
    global_config=GlobalConfig(default_parts=2),
    axis_rules=[AxisRule(op_type="Add", prefer_axis=1)],
)
```

and expects the resulting plan axis to be `1`.

- [ ] **Step 2: Run the test to verify red**

Run: `./.venv/bin/pytest tests/test_splitter_planner.py::test_planner_uses_axis_rule_preference -v`
Expected: FAIL because planner currently defaults to axis `0`.

- [ ] **Step 3: Write a failing precedence test**

Add a test showing operator-specific config wins over `axis_rules`:

```python
SplitConfig(
    operators={"test_op": OperatorConfig(parts=2, axis=0)},
    axis_rules=[AxisRule(op_type="Add", prefer_axis=1)],
)
```

Expected plan axis: `0`.

- [ ] **Step 4: Write a failing `prefer_axis: null` test**

Add a test showing:

```python
SplitConfig(
    global_config=GlobalConfig(default_parts=2),
    axis_rules=[AxisRule(op_type="Add", prefer_axis=None)],
)
```

produces no auto plan for that operator unless an explicit operator config exists.

- [ ] **Step 5: Run the three planner tests to verify red**

Run: `./.venv/bin/pytest tests/test_splitter_planner.py::test_planner_uses_axis_rule_preference tests/test_splitter_planner.py::test_planner_operator_axis_overrides_axis_rule tests/test_splitter_planner.py::test_planner_axis_rule_none_disables_auto_split -v`
Expected: FAIL for missing runtime behavior.

- [ ] **Step 6: Write the minimal planner implementation**

In `src/onnxsplit/splitter/planner.py`:

- add helper logic to find the first matching `AxisRule` by `op_type`
- add helper logic to order candidate axes using:
  - explicit operator axis first
  - `prefer_axis=int`
  - `prefer_axis=="batch"` as axis `0`
  - `prefer_axis is None` as "disable auto axis selection"
- keep current fallback ordering when no rule matches

- [ ] **Step 7: Run the focused planner tests**

Run: `./.venv/bin/pytest tests/test_splitter_planner.py::test_planner_uses_axis_rule_preference tests/test_splitter_planner.py::test_planner_operator_axis_overrides_axis_rule tests/test_splitter_planner.py::test_planner_axis_rule_none_disables_auto_split -v`
Expected: PASS

- [ ] **Step 8: Run the full planner suite**

Run: `./.venv/bin/pytest tests/test_splitter_planner.py -v`
Expected: PASS

### Task 3: Activate `overflow_strategy` In Auto Adjustment

**Files:**
- Modify: `src/onnxsplit/memory/auto_adjust.py`
- Modify: `src/onnxsplit/cli/runner.py`
- Test: `tests/test_auto_adjust.py`

- [ ] **Step 1: Write a failing test for explicit linear strategy**

Add a test in `tests/test_auto_adjust.py` that constructs an operator whose next valid divisible `parts` differs between linear search and binary search, and asserts that linear mode returns the first valid upward candidate.

- [ ] **Step 2: Write a failing test for binary default behavior**

Add a test asserting that existing callers without a strategy still behave like binary mode.

- [ ] **Step 3: Run the new adjuster tests to verify red**

Run: `./.venv/bin/pytest tests/test_auto_adjust.py::test_adjust_plan_linear_strategy_prefers_first_valid_candidate tests/test_auto_adjust.py::test_adjust_plan_defaults_to_binary_strategy -v`
Expected: FAIL because strategy selection does not exist yet.

- [ ] **Step 4: Write the minimal implementation**

In `src/onnxsplit/memory/auto_adjust.py`:

- add a strategy parameter such as:

```python
overflow_strategy: str = "binary_split"
```

- split `_calculate_needed_parts()` into strategy-aware helpers
- preserve divisibility validation and fallback behavior

In `src/onnxsplit/cli/runner.py`:

- read `config.memory_rules.overflow_strategy`
- pass it to the adjuster call
- default to `"binary_split"` when unset

- [ ] **Step 5: Run the focused adjuster tests**

Run: `./.venv/bin/pytest tests/test_auto_adjust.py::test_adjust_plan_linear_strategy_prefers_first_valid_candidate tests/test_auto_adjust.py::test_adjust_plan_defaults_to_binary_strategy -v`
Expected: PASS

- [ ] **Step 6: Run the full auto-adjust suite**

Run: `./.venv/bin/pytest tests/test_auto_adjust.py tests/test_memory_adjust_bug.py tests/test_memory_adjust_invalid_parts.py -v`
Expected: PASS

### Task 4: Align README And User-Facing Examples

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update CLI usage examples**

Document `--config` in the split examples.

- [ ] **Step 2: Update the config example to the real schema**

Use:

```yaml
global:
  default_parts: 2
  max_memory_mb: 1024

operators:
  "/model/Conv_0":
    parts: 4
    axis: 0

axis_rules:
  - op_type: "MatMul"
    prefer_axis: "batch"

memory_rules:
  auto_adjust: true
  overflow_strategy: "binary_split"
```

- [ ] **Step 3: Update requirements text**

Change Python requirement to `3.13+`.

- [ ] **Step 4: Manually review README for consistency**

Check that CLI examples, config field names, and version requirements all match the code.

### Task 5: Final Verification

**Files:**
- Verify only

- [ ] **Step 1: Run the focused changed-area suite**

Run: `./.venv/bin/pytest tests/test_cli_parser.py tests/test_cli_runner.py tests/test_splitter_planner.py tests/test_auto_adjust.py -q`
Expected: PASS

- [ ] **Step 2: Run the full project suite**

Run: `./.venv/bin/pytest -q`
Expected: PASS

- [ ] **Step 3: Inspect diff before handoff**

Run: `git diff -- src/onnxsplit/cli/parser.py src/onnxsplit/cli/runner.py src/onnxsplit/splitter/planner.py src/onnxsplit/memory/auto_adjust.py tests/test_cli_parser.py tests/test_cli_runner.py tests/test_splitter_planner.py tests/test_auto_adjust.py README.md docs/superpowers/specs/2026-03-20-cli-config-and-rule-activation-design.md docs/superpowers/plans/2026-03-20-cli-config-and-rule-activation.md`
Expected: Only the planned files are changed.

- [ ] **Step 4: Commit**

```bash
git add README.md \
  src/onnxsplit/cli/parser.py \
  src/onnxsplit/cli/runner.py \
  src/onnxsplit/splitter/planner.py \
  src/onnxsplit/memory/auto_adjust.py \
  tests/test_cli_parser.py \
  tests/test_cli_runner.py \
  tests/test_splitter_planner.py \
  tests/test_auto_adjust.py \
  docs/superpowers/specs/2026-03-20-cli-config-and-rule-activation-design.md \
  docs/superpowers/plans/2026-03-20-cli-config-and-rule-activation.md
git commit -m "feat: activate CLI config and rule-driven split behavior"
```
