from typing import Any

import pandas as pd

from astronaut.logics.quantum_kernel.feedback import FeedbackFormatter
from astronaut.schema import GeneratedResults


def test_feedback_formatter(
    mocker: Any,
    mock_generated_results: GeneratedResults,
    trial_result_df: pd.DataFrame,
    raw_result_df: pd.DataFrame,
) -> None:
    mocker.patch(
        "astronaut.logics.quantum_kernel.utils.get_primary_metric_info",
        return_value=("mean_absolute_error", True),
    )
    mocker.patch(
        "astronaut.logics.quantum_kernel.utils.get_default_metrics_from_config",
        return_value=["mean_absolute_error", "r2_score"],
    )

    formatter = FeedbackFormatter(
        generated_results=mock_generated_results,
        trial_result_df=trial_result_df,
        raw_result_df=raw_result_df,
    )

    feedback_text = formatter.format()

    assert "Idea Number: 1" in feedback_text
    assert "Idea Explanation:" in feedback_text
    assert "E1" in feedback_text  # idea explanation from fixture
    assert "Idea Mathematical Expression:" in feedback_text
    assert "F1" in feedback_text  # idea formula
    assert "Idea Implementation:" in feedback_text
    assert "code1" in feedback_text  # idea code
    assert "Run Time: 10.00 [seconds]" in feedback_text
    assert "Aggregated mean_absolute_error: 0.1" in feedback_text

    assert "Dataset: iris" in feedback_text
    assert "- mean_absolute_error: 0.05" in feedback_text
    assert "- r2_score: 0.95" in feedback_text
    assert "Dataset: digits" in feedback_text
    assert "- mean_absolute_error: 0.15" in feedback_text
    assert "- r2_score: 0.85" in feedback_text

    assert (
        "Current trial ideas sorted by mean_absolute_error from lowest to highest"
        in feedback_text
    )


def test_feedback_formatter_maximize(
    mocker: Any,
    mock_generated_results: GeneratedResults,
    trial_result_df: pd.DataFrame,
    raw_result_df: pd.DataFrame,
) -> None:
    mocker.patch(
        "astronaut.logics.quantum_kernel.utils.get_primary_metric_info",
        return_value=("accuracy", False),
    )
    mocker.patch(
        "astronaut.logics.quantum_kernel.utils.get_default_metrics_from_config",
        return_value=["accuracy"],
    )

    trial_result_df_acc = trial_result_df.copy()
    trial_result_df_acc.rename(
        columns={
            "aggregated_mean_absolute_error_validation": "aggregated_accuracy_validation",
            "aggregated_mean_absolute_error_test": "aggregated_accuracy_test",
        },
        inplace=True,
    )

    raw_result_df_acc = raw_result_df.copy()
    raw_result_df_acc["accuracy"] = [0.95, 0.85]

    formatter = FeedbackFormatter(
        generated_results=mock_generated_results,
        trial_result_df=trial_result_df_acc,
        raw_result_df=raw_result_df_acc,
    )

    feedback_text = formatter.format()

    assert (
        "Current trial ideas sorted by accuracy from highest to lowest"
        in feedback_text
    )
    assert "Aggregated accuracy: 0.1" in feedback_text
