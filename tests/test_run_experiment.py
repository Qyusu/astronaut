import pandas as pd
import pytest

from astronaut.run_experiment import update_run_context
from astronaut.schema import (
    MessageHistory,
    MessageHistoryNum,
    ModelVersions,
    RunContext,
)


@pytest.fixture(scope="function")
def run_context(model_versions: ModelVersions) -> RunContext:
    return RunContext(
        gen_config_dirc="configs.generated",
        gen_code_dirc="astronaut.generated",
        model_versions=model_versions,
        n_qubits=10,
        max_trial_num=30,
        max_idea_num=2,
        max_suggestion_num=3,
        max_reflection_round=3,
        best_idea_abstract="This is a test idea explanation.",
        last_code="This is a test code.",
        score_list=[],
        total_cost=0.0,
        need_idea_review=False,
        review_comment="",
        last_trial_results="",
        score_histories="",
        message_history=MessageHistory(),
        n_message_history=MessageHistoryNum(),
        eval_result_df=pd.DataFrame(),
    )


class TestRunContext:
    def test_update_value(self, run_context: RunContext) -> None:
        updated_code = "This is an updated test code."
        updated_context = update_run_context(run_context, last_code=updated_code)
        assert updated_context.last_code == updated_code

    def test_update_class_instance(self, run_context: RunContext) -> None:
        updated_n_message_history = MessageHistoryNum(
            review=1,
            idea=2,
            code=3,
        )
        updated_context = update_run_context(run_context, n_message_history=updated_n_message_history)
        assert updated_context.n_message_history == updated_n_message_history

    def test_update_dataframe(self, run_context: RunContext) -> None:
        updated_df = pd.DataFrame({"test": [1, 2, 3]})
        updated_context = update_run_context(run_context, eval_result_df=updated_df)
        assert updated_context.eval_result_df.equals(updated_df)

    def test_update_value_with_invalid_key(self, run_context: RunContext) -> None:
        with pytest.raises(ValueError):
            update_run_context(run_context, invalid_key="This is an invalid key.")
