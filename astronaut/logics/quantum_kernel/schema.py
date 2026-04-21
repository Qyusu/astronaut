from pydantic import BaseModel, ConfigDict


class QuantumKernelExperimentResult(BaseModel):
    """Schema for single dataset experiment result."""

    model_config = ConfigDict(extra="allow")

    idea_id: str
    dataset_name: str
    run_time: float
    # Other dynamic metric fields will be included via extra="allow"


class QuantumKernelAggregatedResult(BaseModel):
    """Schema for aggregated experiment result (across datasets)."""

    model_config = ConfigDict(extra="allow")

    idea_id: str
    aggregated_run_time: float
    # Other dynamic metric fields (aggregated_*) will be included via extra="allow"


class FeedbackTemplateVariables(BaseModel):
    """Schema for variables used in feedback template."""

    idea_num: int
    idea_explanation: str
    idea_formula: str
    idea_implementation: str
    run_time: float
    aggregated_metric_label: str
    aggregated_metric_value: float
    raw_result_str: str


class HyperparameterSearchArguments(BaseModel):
    """Schema for hyperparameter search arguments."""

    cv: int
    direction: str
    n_jobs: int
    n_trials: int
