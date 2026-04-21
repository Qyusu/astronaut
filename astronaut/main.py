import sys

import qxmt

import astronaut.constants as exp_const
from astronaut.configs import ExperimentSettings
from astronaut.initializers import (
    initialize_algorithm_context,
    initialize_clients,
    initialize_logger,
    load_dataset_configs,
)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m astronaut.main <EXPERIMENT_CONFIG_PATH.yaml>"
        )

    config_path = sys.argv[1]
    experiment_settings = ExperimentSettings.from_yaml(config_path)
    dataset_configs = load_dataset_configs(config_path, experiment_settings)

    initialize_logger(
        experiment_settings.experiment_name,
        experiment_settings.console_log_level,
        experiment_settings.log_level,
    )

    chat_client, _, qml_db_client, arxiv_db_client = initialize_clients()

    experiment = qxmt.Experiment(
        name=experiment_settings.experiment_name,
        desc=experiment_settings.desc,
        auto_gen_mode=False,
    ).init()

    context = initialize_algorithm_context(
        experiment_settings.task_type,
        experiment_settings,
        chat_client,
        arxiv_db_client,
        qml_db_client,
        experiment,
        dataset_configs,
    )

    context.execute_baseline(
        exp_const.BASE_CONFIG_PATH.replace(
            exp_const.TASK_TYPE_PLACEHOLDER, experiment_settings.task_type
        )
    )
    context.execute_experiment()


if __name__ == "__main__":
    main()
