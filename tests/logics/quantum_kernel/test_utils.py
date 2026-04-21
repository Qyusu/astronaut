from typing import Any

import pytest
import yaml

import astronaut.logics.quantum_kernel.constants as qk_const
from astronaut.logics.quantum_kernel.utils import (
    get_dataset_config,
    get_default_metrics_from_config,
    get_n_qubits_from_config,
    get_primary_metric_info,
)


def test_get_dataset_config() -> None:
    dataset_configs: dict[str, Any] = {
        "iris": {"n_samples": 150},
        "digits": {"n_samples": 1797},
    }

    # Success cases
    assert get_dataset_config("iris", dataset_configs) == {"n_samples": 150}
    assert get_dataset_config("digits", dataset_configs) == {"n_samples": 1797}

    # Dataset not provided
    with pytest.raises(
        ValueError, match="Dataset configuration is not provided"
    ):
        get_dataset_config("iris", None)

    # Dataset not found
    with pytest.raises(ValueError, match="Dataset 'wine' not found"):
        get_dataset_config("wine", dataset_configs)


def test_get_n_qubits_from_config(mocker: Any) -> None:
    mock_yaml: dict[str, Any] = {
        qk_const.QXMT_CONFIG_KEY_DEVICE: {qk_const.QXMT_CONFIG_KEY_N_QUBITS: 4}
    }
    mock_file_content = yaml.dump(mock_yaml)

    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))
    assert get_n_qubits_from_config("dummy_path.yaml") == 4

    # Case: n_qubits not found
    mock_yaml_invalid: dict[str, Any] = {qk_const.QXMT_CONFIG_KEY_DEVICE: {}}
    mock_file_content_invalid = yaml.dump(mock_yaml_invalid)
    mocker.patch(
        "builtins.open", mocker.mock_open(read_data=mock_file_content_invalid)
    )
    with pytest.raises(ValueError, match="n_qubits is not found"):
        get_n_qubits_from_config("dummy_path.yaml")


def test_get_primary_metric_info(mocker: Any) -> None:
    mock_yaml: dict[str, Any] = {
        qk_const.QXMT_CONFIG_KEY_EVALUATION: {
            qk_const.QXMT_CONFIG_KEY_DEFAULT_METRICS: [
                "mean_absolute_error",
                "r2_score",
            ]
        }
    }
    mock_file_content = yaml.dump(mock_yaml)

    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))
    metric, is_minimize = get_primary_metric_info()
    assert metric == "mean_absolute_error"
    assert is_minimize is True


def test_get_default_metrics_from_config(mocker: Any) -> None:
    mock_yaml: dict[str, Any] = {
        qk_const.QXMT_CONFIG_KEY_EVALUATION: {
            qk_const.QXMT_CONFIG_KEY_DEFAULT_METRICS: ["accuracy", "f1"]
        }
    }
    mock_file_content = yaml.dump(mock_yaml)

    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))
    metrics = get_default_metrics_from_config()
    assert metrics == ["accuracy", "f1"]
