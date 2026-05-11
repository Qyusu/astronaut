import numpy as np
import pandas as pd


class MultiCostAggregator:
    def __init__(self, result_df: pd.DataFrame) -> None:
        self.result_df = result_df

    def aggregate(self, aggregate_method: str, cost_col: str = "cost") -> float:
        match aggregate_method:
            case "min":
                return self._aggregate_by_min(cost_col)
            case "max":
                return self._aggregate_by_max(cost_col)
            case "mean":
                return self._aggregate_by_mean(cost_col)
            case "median":
                return self._aggregate_by_median(cost_col)
            case _:
                raise ValueError(
                    f"Invalid aggregate method: {aggregate_method}"
                )

    def _aggregate_by_min(self, cost_col: str) -> float:
        return float(self.result_df[cost_col].min())

    def _aggregate_by_max(self, cost_col: str) -> float:
        return float(self.result_df[cost_col].max())

    def _aggregate_by_mean(self, cost_col: str) -> float:
        return float(self.result_df[cost_col].mean())

    def _aggregate_by_median(self, cost_col: str) -> float:
        return float(self.result_df[cost_col].median())


def cost_function(final_energy: float, exact_energy: float) -> float:
    return np.abs(final_energy - exact_energy)
