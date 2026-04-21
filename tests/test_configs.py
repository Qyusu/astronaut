import pytest
import yaml
from pydantic import ValidationError
from pytest_mock import MockerFixture

import astronaut.constants as exp_const
from astronaut.configs import ExperimentSettings, Settings
from astronaut.schema import ReflectionType, ResearchMode


class TestExperimentSettings:
    def test_valid_init(self) -> None:
        settings = ExperimentSettings(
            task_type="quantum_kernel", experiment_name="test_exp"
        )
        assert settings.task_type == "quantum_kernel"
        assert settings.experiment_name == "test_exp"
        assert settings.dataset_config_path == "configs/datasets.yaml"
        assert settings.max_trial_num == exp_const.DEFAULT_MAX_TRIAL_NUM
        assert settings.research_mode == ResearchMode.NONE
        assert settings.reflection_type == ReflectionType.NONE

    def test_invalid_task_type(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentSettings(task_type="invalid_type", experiment_name="test")

    def test_from_yaml(self, mocker: MockerFixture) -> None:
        yaml_content = """
        task_type: "vqe"
        experiment_name: "yaml_exp"
        max_trial_num: 10
        """
        # mock open and yaml.safe_load using mocker
        mocker.patch("builtins.open", mocker.mock_open(read_data=yaml_content))
        mocker.patch(
            "yaml.safe_load", return_value=yaml.safe_load(yaml_content)
        )

        settings = ExperimentSettings.from_yaml("dummy_path.yaml")

        assert settings.task_type == "vqe"
        assert settings.experiment_name == "yaml_exp"
        assert settings.max_trial_num == 10


class TestSettings:
    def test_default_model_versions(self) -> None:
        settings = Settings(
            DEFAULT_MODEL_VERSION="gpt-5",
            DEFAULT_LIGHT_MODEL_VERSION="gpt-5-nano",
            # Pass empty strings for others to trigger __init__ logic
            EXPANDING_MODEL_VERSION="",
            IDEA_MODEL_VERSION="",
            SCORING_MODEL_VERSION="",
            SUMMARY_MODEL_VERSION="",
            EXPERT_MODEL_VERSION="",
            REFLECTION_MODEL_VERSION="",
            REFERENCE_EXTRACTION_MODEL_VERSION="",
            CODE_MODEL_VERSION="",
            REVIEW_MODEL_VERSION="",
            # Disable loading from .env for this test if possible, or just override everything
            _env_file=None,
        )

        assert settings.EXPANDING_MODEL_VERSION == "gpt-5"
        assert settings.IDEA_MODEL_VERSION == "gpt-5"
        assert settings.SCORING_MODEL_VERSION == "gpt-5-nano"
        assert settings.SUMMARY_MODEL_VERSION == "gpt-5-nano"
        assert settings.EXPERT_MODEL_VERSION == "gpt-5-nano"
        assert settings.REFLECTION_MODEL_VERSION == "gpt-5"
        assert settings.REFERENCE_EXTRACTION_MODEL_VERSION == "gpt-5-nano"
        assert settings.CODE_MODEL_VERSION == "gpt-5"
        assert settings.REVIEW_MODEL_VERSION == "gpt-5"

    def test_none_pattern_handling(self) -> None:
        settings = Settings(
            PINECONE_API_KEY="None",
            PENNLYLANE_INDEX_NAME="NONE",
            ARXIV_INDEX_NAME="none",
            EMBEDDING_MODEL_VERSION="",
            _env_file=None,
        )
        assert settings.PINECONE_API_KEY is None
        assert settings.PENNLYLANE_INDEX_NAME is None
        assert settings.ARXIV_INDEX_NAME is None
        assert settings.EMBEDDING_MODEL_VERSION is None

    def test_valid_optional_values(self) -> None:
        settings = Settings(
            PINECONE_API_KEY="valid_key",
            PENNLYLANE_INDEX_NAME="valid_index",
            ARXIV_INDEX_NAME="valid_arxiv",
            EMBEDDING_MODEL_VERSION="valid_model",
            _env_file=None,
        )
        assert settings.PINECONE_API_KEY == "valid_key"
        assert settings.PENNLYLANE_INDEX_NAME == "valid_index"
        assert settings.ARXIV_INDEX_NAME == "valid_arxiv"
        assert settings.EMBEDDING_MODEL_VERSION == "valid_model"
