# CLI Config And Rule Activation Design

## Goal

Make the existing configuration system actually usable from the CLI and ensure loaded rule fields participate in runtime behavior instead of being dead schema.

## Scope

This design covers four changes:

1. Add `--config` to the `split` command and pass it through to the existing config loader path.
2. Activate `axis_rules` during split planning when an operator does not have an explicit axis configured.
3. Activate `memory_rules.overflow_strategy` during automatic part adjustment.
4. Align user-facing documentation with the real schema and runtime behavior.

Out of scope:

- New config schema fields
- New operator split heuristics beyond existing analyzers
- Deep redesign of graph transformation behavior

## Current State

The project already has:

- `RunContext.config_path` and `run_split()` support for loading config files
- `SplitConfig.axis_rules` and `SplitConfig.memory_rules.overflow_strategy` in the schema
- YAML loader support for `global`, `operators`, `axis_rules`, and `memory_rules`

The current gaps are:

- The `split` CLI command does not expose `--config`, so config loading is inaccessible from normal CLI usage.
- `SplitPlanner` instantiates `AxisAnalyzer()` but does not consume `config.axis_rules`.
- `AutoSplitAdjuster` always uses the current binary-search behavior and does not consume `overflow_strategy`.
- The README shows an outdated config example that does not match the loader schema.

## Design Decisions

### 1. CLI Config Wiring

Add an optional `--config` option to `onnxsplit split`.

Behavior:

- When omitted, behavior stays unchanged.
- When provided, the CLI passes the path into `RunContext.config_path`.
- `run_split()` continues to load and merge config through the existing `_load_config()` and `_prepare_config()` flow.

This is intentionally minimal because the runner path already exists and is covered by tests.

### 2. Axis Rule Activation

Keep responsibilities separate:

- `AxisAnalyzer.analyze()` continues to answer "which axes are splitable for this operator?"
- `SplitPlanner` decides "which candidate axis should be tried first?"

`axis_rules` will only affect axis preference ordering when the operator does not already have an explicit axis in `operators[...]`.

Resolution order:

1. Operator-specific `axis` from `operators[...]`
2. Matching `axis_rules` entry for the operator's `op_type`
3. Existing default preference: axis `0` first, then ascending numeric order

Supported `prefer_axis` behavior:

- Integer: prefer that axis if it is splitable, then try the remaining splitable axes.
- `"batch"`: alias for axis `0`.
- `null`: disable automatic split planning for that operator type unless the operator has an explicit axis in `operators[...]`.

Matching behavior:

- Match on exact `op_type`.
- First matching rule wins.

Why `null` means "disable automatic split planning":

- The current schema already represents a deliberate "no default axis" signal.
- Without this behavior, `null` becomes effectively meaningless at runtime.
- Explicit operator config should still override it because explicit per-operator config is more specific than type-level preference.

### 3. Overflow Strategy Activation

`memory_rules.overflow_strategy` will control how `AutoSplitAdjuster` searches for a satisfying `parts` value once memory reduction is needed.

Supported strategies:

- `binary_split`: keep the current binary-search approach to find the smallest satisfying candidate quickly.
- `linear_split`: linearly search upward from the validated lower bound until the first satisfying candidate is found.

Common rules for both strategies:

- Returned `parts` must still pass divisibility validation against the selected axis.
- If no valid `parts` can satisfy the constraint, preserve current fallback behavior to `parts=1`.
- If no strategy is configured, default to `binary_split` to preserve current behavior.

Implementation boundary:

- The strategy selection should be explicit in runner-to-adjuster flow or adjuster API, rather than hidden global state.
- Existing callers without a strategy should continue to work.

### 4. Documentation Alignment

Update the README so it reflects the real supported interface:

- Config example uses `global`, `operators`, `axis_rules`, `memory_rules`
- CLI examples include `--config`
- Python version requirement matches `pyproject.toml` (`>=3.13`)

## Files Affected

- `src/onnxsplit/cli/parser.py`
- `src/onnxsplit/cli/runner.py`
- `src/onnxsplit/splitter/planner.py`
- `src/onnxsplit/memory/auto_adjust.py`
- `tests/test_cli_parser.py`
- `tests/test_cli_runner.py`
- `tests/test_splitter_planner.py`
- `tests/test_auto_adjust.py`
- `README.md`

## Testing Strategy

Use TDD for each behavior change.

Add tests for:

- `split --config ...` parsing and end-to-end use
- axis rule preference selection
- operator-specific axis overriding type rules
- `prefer_axis: null` disabling automatic planning
- binary vs linear overflow strategy behavior
- unchanged defaults when config fields are absent

Regression goal:

- Existing suite should remain green
- New behavior should be covered by focused unit tests rather than only broad integration tests

## Risks

1. `axis_rules` could accidentally override explicit operator config.
   Mitigation: encode precedence in tests first.

2. `linear_split` could return a different valid `parts` than binary mode and break assumptions in existing tests.
   Mitigation: add explicit tests for each strategy and keep the default as binary.

3. README updates could still drift from code.
   Mitigation: keep examples tied to the actual loader schema already exercised by fixture configs.
