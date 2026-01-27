"""配置合并逻辑：将命令行参数与配置文件合并"""

from dataclasses import replace

from onnxsplit.config.schema import SplitConfig


class ConfigMergeError(Exception):
    """配置合并错误"""


def merge_cli_args(
    config: SplitConfig,
    cli_parts: int | None,
    cli_max_memory: float | None,
) -> SplitConfig:
    """合并CLI参数到配置中。

    Args:
        config: 原始配置对象
        cli_parts: CLI指定的parts参数，None表示不覆盖
        cli_max_memory: CLI指定的max_memory参数，None表示不覆盖

    Returns:
        合并后的新配置对象。如果两个CLI参数都为None，返回原配置对象。

    Raises:
        ConfigMergeError: 当CLI参数值无效时
    """
    # 如果两个CLI参数都为None，返回原配置
    if cli_parts is None and cli_max_memory is None:
        return config

    # 验证cli_parts
    if cli_parts is not None:
        if cli_parts <= 0:
            raise ConfigMergeError(f"cli_parts必须大于0，得到: {cli_parts}")

    # 验证cli_max_memory
    if cli_max_memory is not None:
        if cli_max_memory <= 0:
            raise ConfigMergeError(f"cli_max_memory必须大于0，得到: {cli_max_memory}")

    # 创建新的GlobalConfig
    new_global = replace(
        config.global_config,
        default_parts=cli_parts if cli_parts is not None else config.global_config.default_parts,
        max_memory_mb=cli_max_memory
        if cli_max_memory is not None
        else config.global_config.max_memory_mb,
    )

    # 创建并返回新的SplitConfig，保留其他配置
    return replace(config, global_config=new_global)
