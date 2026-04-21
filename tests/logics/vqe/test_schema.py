from astronaut.logics.vqe.schema import (
    FeedbackTemplateVariables,
    VQEAggregatedResult,
    VQEExperimentResult,
)


def test_vqe_experiment_result_schema() -> None:
    result = VQEExperimentResult(
        idea_id="1_1",
        molecule_name="H2",
        num_iter=100,
        run_time=0.5,
        bond_length=0.74,
        cost=0.01,
    )
    assert result.idea_id == "1_1"
    assert result.molecule_name == "H2"


def test_vqe_aggregated_result_schema() -> None:
    result = VQEAggregatedResult(
        idea_id="1_1",
        num_gates_order="O(N)",
        num_params_order="O(N)",
        aggregated_num_iter=100.0,
        aggregated_run_time=0.5,
        aggregated_cost=0.01,
    )
    assert result.idea_id == "1_1"
    assert VQEAggregatedResult.aggregation_source()["aggregated_cost"] == "cost"


def test_feedback_template_variables_schema() -> None:
    vars = FeedbackTemplateVariables(
        idea_num=1,
        idea_explanation="Expl",
        idea_formula="Form",
        num_gates_order="O(N)",
        num_params_order="O(N)",
        run_time=0.5,
        num_iter=100.0,
        cost=0.01,
        molecule_bond_length_results_str="res",
    )
    assert vars.idea_num == 1
    assert vars.molecule_bond_length_results_str == "res"
