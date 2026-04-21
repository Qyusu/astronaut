from typing import Any

import pandas as pd
import pytest

import astronaut.logics.quantum_kernel.constants as qk_const
import astronaut.schema as base_schema


@pytest.fixture
def mock_experiment(mocker: Any) -> Any:
    return mocker.MagicMock()


@pytest.fixture
def mock_llm_client(mocker: Any) -> Any:
    return mocker.MagicMock()


@pytest.fixture
def run_context() -> base_schema.RunContext:
    model_versions = base_schema.ModelVersions(
        default="test-default",
        expanding="test-expanding",
        idea="test-idea",
        scoring="test-scoring",
        summary="test-summary",
        expert="test-expert",
        reflection="test-reflection",
        reference="test-ref",
        code="test-code",
        review="test-review",
    )
    return base_schema.RunContext(
        gen_config_dirc="test_configs",
        gen_code_dirc="test_code",
        local_paper_dir="test_papers",
        model_versions=model_versions,
        max_trial_num=3,
        max_idea_num=1,
        max_suggestion_num=5,
        max_scoring_round=1,
        reflection_config=base_schema.ReflectionConfig(
            type=base_schema.ReflectionType.SELF_REFLECTION
        ),
        last_code="print(1)",
        total_cost=0.0,
        need_idea_review=False,
        review_comment="None",
        last_trial_results="None",
        score_histories="None",
        message_history=base_schema.MessageHistory(),
        n_message_history=base_schema.MessageHistoryNum(),
        eval_result_df=pd.DataFrame(),
    )


@pytest.fixture
def dataset_configs() -> dict[str, Any]:
    return {
        "iris": {"n_samples": 100, "n_features": 4, "n_classes": 3},
        "digits": {"n_samples": 200, "n_features": 64, "n_classes": 10},
    }


@pytest.fixture
def mock_generated_results() -> base_schema.GeneratedResults:
    idea1 = base_schema.GeneratedIdea(
        name="Idea 1",
        summary="S1",
        explanation="E1",
        formula="F1",
        key_sentences=["K1"],
    )
    res1 = base_schema.GeneratedResult(
        idea=idea1,
        score=None,
        implement=base_schema.GeneratedImpl(
            class_name="FeatureMap1", code="code1"
        ),
    )
    return base_schema.GeneratedResults(results=[res1])


@pytest.fixture
def trial_result_df() -> pd.DataFrame:
    metric = "mean_absolute_error"
    return pd.DataFrame(
        [
            {
                qk_const.IDEA_ID_COL: "1_1",
                qk_const.AGGREGATED_RUN_TIME_COL: 10.0,
                f"{qk_const.AGGREGATED_COL_PREFIX}_{metric}_{qk_const.VALIDATION_SUFFIX}": 0.1,
                f"{qk_const.AGGREGATED_COL_PREFIX}_{metric}_{qk_const.TEST_SUFFIX}": 0.15,
            }
        ]
    )


@pytest.fixture
def raw_result_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                qk_const.IDEA_ID_COL: "1_1",
                qk_const.DATASET_NAME_COL: "iris",
                "mean_absolute_error": 0.05,
                "r2_score": 0.95,
            },
            {
                qk_const.IDEA_ID_COL: "1_1",
                qk_const.DATASET_NAME_COL: "digits",
                "mean_absolute_error": 0.15,
                "r2_score": 0.85,
            },
        ]
    )
