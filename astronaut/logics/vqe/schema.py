from pydantic import BaseModel, ConfigDict


class VQEExperimentResult(BaseModel):
    """Schema for single VQE experiment result (one bond length)."""

    model_config = ConfigDict(extra="allow")

    idea_id: str
    molecule_name: str
    num_iter: int
    run_time: float
    bond_length: float
    cost: float


class VQEAggregatedResult(BaseModel):
    """Schema for aggregated VQE experiment result (across bond lengths)."""

    idea_id: str
    num_gates_order: str
    num_params_order: str
    aggregated_num_iter: float
    aggregated_run_time: float
    aggregated_cost: float

    @classmethod
    def aggregation_source(cls) -> dict[str, str]:
        """Mapping from aggregation field name to source field name in VQEExperimentResult."""
        return {
            "aggregated_num_iter": "num_iter",
            "aggregated_run_time": "run_time",
            "aggregated_cost": "cost",
        }


class FeedbackTemplateVariables(BaseModel):
    """Schema for feedback template variables."""

    idea_num: int
    idea_explanation: str
    idea_formula: str
    num_gates_order: str
    num_params_order: str
    run_time: float
    num_iter: float
    cost: float
    molecule_bond_length_results_str: str = ""
