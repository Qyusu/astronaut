import pandas as pd
import pytest

from astronaut.schema import (
    MessageHistory,
    MessageHistoryNum,
    ModelVersions,
    ReflectionConfig,
    ReflectionType,
    RunContext,
)


@pytest.fixture(scope="function")
def run_context(model_versions: ModelVersions) -> RunContext:
    return RunContext(
        model_type="quantum_kernel",
        gen_config_dirc="configs.generated",
        gen_code_dirc="astronaut.generated",
        local_paper_dir="papers/arxiv_index",
        model_versions=model_versions,
        n_qubits=10,
        max_trial_num=30,
        max_idea_num=2,
        max_suggestion_num=3,
        max_scoring_round=2,
        reflection_config=ReflectionConfig(
            type=ReflectionType.SELF_REFLECTION, max_rounds=3
        ),
        best_idea_abstract="This is a test idea explanation.",
        last_code="This is a test code.",
        total_cost=0.0,
        need_idea_review=False,
        review_comment="",
        last_trial_results="",
        score_histories="",
        message_history=MessageHistory(),
        n_message_history=MessageHistoryNum(),
        eval_result_df=pd.DataFrame(),
    )
