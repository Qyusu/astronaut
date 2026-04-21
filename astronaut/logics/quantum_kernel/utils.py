from typing import Any

import yaml

import astronaut.constants as exp_const
import astronaut.logics.quantum_kernel.constants as qk_const

QKERNEL_BASE_CONFIG_PATH = exp_const.BASE_CONFIG_PATH.replace(
    exp_const.TASK_TYPE_PLACEHOLDER, exp_const.QKERNEL_MODEL_TYPE
)


def get_dataset_config(
    dataset_name: str, dataset_configs: dict[str, Any] | None = None
) -> dict[str, Any]:
    if dataset_configs is None:
        raise ValueError(
            "Dataset configuration is not provided via dataset_configs. "
            "Please ensure dataset_config_path is set in the experiment configuration."
        )

    if dataset_name not in dataset_configs:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in loaded configurations. "
            f"Available datasets: {list(dataset_configs.keys())}"
        )

    return dataset_configs[dataset_name]


def get_n_qubits_from_config(config_path: str) -> int:
    config = yaml.safe_load(open(config_path))
    n_qubits = config.get(qk_const.QXMT_CONFIG_KEY_DEVICE, {}).get(
        qk_const.QXMT_CONFIG_KEY_N_QUBITS, None
    )
    if n_qubits is None:
        raise ValueError("n_qubits is not found in the config.")
    return n_qubits


def get_primary_metric_info() -> tuple[str, bool]:
    """Get primary metric and its optimization direction (minimize or not)."""
    config = yaml.safe_load(open(QKERNEL_BASE_CONFIG_PATH))
    primary_metric = config[qk_const.QXMT_CONFIG_KEY_EVALUATION][
        qk_const.QXMT_CONFIG_KEY_DEFAULT_METRICS
    ][0]
    direction = qk_const.METRIC_OPTIMIZATION_DIRECTIONS.get(
        primary_metric, "maximize"
    )
    is_minimize = direction == "minimize"
    return primary_metric, is_minimize


def get_default_metrics_from_config() -> list[str]:
    """Get list of default metrics from the configuration."""
    config = yaml.safe_load(open(QKERNEL_BASE_CONFIG_PATH))
    return list(
        config[qk_const.QXMT_CONFIG_KEY_EVALUATION][
            qk_const.QXMT_CONFIG_KEY_DEFAULT_METRICS
        ]
    )
