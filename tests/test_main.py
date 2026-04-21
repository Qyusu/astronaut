import sys

import pytest
from pytest_mock import MockerFixture

from astronaut.main import main


def test_main_execution(mocker: MockerFixture) -> None:
    # Mock sys.argv
    mocker.patch.object(sys, "argv", ["main.py", "experiment.yaml"])

    # Mock Configs
    mock_settings = mocker.Mock()
    mock_settings.experiment_name = "test_exp"
    mock_settings.desc = "desc"
    mock_settings.task_type = "quantum_kernel"
    mocker.patch(
        "astronaut.main.ExperimentSettings.from_yaml",
        return_value=mock_settings,
    )

    # Mock Initializers
    mocker.patch("astronaut.main.load_dataset_configs", return_value={})
    mocker.patch("astronaut.main.initialize_logger")
    mock_chat = mocker.Mock()
    mock_qml = mocker.Mock()
    mock_arxiv = mocker.Mock()
    mocker.patch(
        "astronaut.main.initialize_clients",
        return_value=(mock_chat, None, mock_qml, mock_arxiv),
    )

    # Mock QXMT
    mock_exp_class = mocker.patch("astronaut.main.qxmt.Experiment")
    mock_exp_instance = mock_exp_class.return_value
    mock_exp_instance.init.return_value

    # Mock Context
    mock_context_func = mocker.patch(
        "astronaut.main.initialize_algorithm_context"
    )
    mock_context = mock_context_func.return_value

    # Run main
    main()
    mock_context.execute_baseline.assert_called_once()
    mock_context.execute_experiment.assert_called_once()


def test_main_missing_args(mocker: MockerFixture) -> None:
    mocker.patch.object(sys, "argv", ["main.py"])
    with pytest.raises(SystemExit, match="Usage: python -m astronaut.main"):
        main()
