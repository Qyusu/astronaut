from astronaut.logics.quantum_kernel.schema import (
    FeedbackTemplateVariables,
    HyperparameterSearchArguments,
    QuantumKernelAggregatedResult,
    QuantumKernelExperimentResult,
)


def test_quantum_kernel_experiment_result() -> None:
    data = {
        "idea_id": "1_1",
        "dataset_name": "iris",
        "run_time": 10.5,
        "accuracy": 0.95,  # Extra field
    }
    obj = QuantumKernelExperimentResult(**data)
    assert obj.idea_id == "1_1"
    assert obj.dataset_name == "iris"
    assert obj.run_time == 10.5
    assert getattr(obj, "accuracy") == 0.95
    assert obj.model_dump()["accuracy"] == 0.95


def test_quantum_kernel_aggregated_result() -> None:
    data = {
        "idea_id": "1_1",
        "aggregated_run_time": 21.0,
        "aggregated_accuracy_validation": 0.9,  # Extra field
    }
    obj = QuantumKernelAggregatedResult(**data)
    assert obj.idea_id == "1_1"
    assert obj.aggregated_run_time == 21.0
    assert getattr(obj, "aggregated_accuracy_validation") == 0.9


def test_feedback_template_variables() -> None:
    data = {
        "idea_num": 1,
        "idea_explanation": "exp",
        "idea_formula": "form",
        "idea_implementation": "impl",
        "run_time": 10.0,
        "aggregated_metric_label": "MAE",
        "aggregated_metric_value": 0.1,
        "raw_result_str": "raw",
    }
    obj = FeedbackTemplateVariables(**data)
    assert obj.idea_num == 1
    assert obj.idea_explanation == "exp"


def test_hyperparameter_search_arguments() -> None:
    data = {
        "cv": 5,
        "direction": "maximize",
        "n_jobs": -1,
        "n_trials": 100,
    }
    obj = HyperparameterSearchArguments(**data)
    assert obj.cv == 5
    assert obj.direction == "maximize"
    assert obj.n_jobs == -1
    assert obj.n_trials == 100
