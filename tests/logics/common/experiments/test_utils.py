import pandas as pd

from astronaut.logics.common.experiments.utils import get_best_result_id, update_run_context
from astronaut.schema import RunContext


def test_update_run_context(run_context: RunContext):
    updated_context = update_run_context(
        run_context,
        n_qubits=5,
        eval_result_df=pd.DataFrame({"idea_id": ["1", "2", "3"], "aggregated_cost": [0.1, 0.05, 0.2]}),
    )

    assert updated_context.n_qubits == 5
    assert updated_context.eval_result_df.equals(
        pd.DataFrame({"idea_id": ["1", "2", "3"], "aggregated_cost": [0.1, 0.05, 0.2]})
    )


def test_get_best_result_id():
    # maximize accuracy (Quantum Kernel)
    result_df = pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "accuracy_validation": [0.5, 0.35, 0.9]})
    best_result_id = get_best_result_id(result_df, "accuracy_validation", minimize=False)
    assert best_result_id == "1_3"

    result_df = pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "accuracy_validation": [0.0, 0.5, 0.0]})
    best_result_id = get_best_result_id(result_df, "accuracy_validation", minimize=False)
    assert best_result_id == "1_2"

    # minimize cost (VQE)
    result_df = pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "aggregated_cost": [0.1, 0.05, 0.2]})
    best_result_id = get_best_result_id(result_df, "aggregated_cost", minimize=True)
    assert best_result_id == "1_2"

    result_df = pd.DataFrame({"idea_id": ["1_1", "1_2", "1_3"], "aggregated_cost": [0.1, float("inf"), float("inf")]})
    best_result_id = get_best_result_id(result_df, "aggregated_cost", minimize=True)
    assert best_result_id == "1_1"
