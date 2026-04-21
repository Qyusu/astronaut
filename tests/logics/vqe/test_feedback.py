import pandas as pd
import pytest

import astronaut.schema as base_schema
from astronaut.logics.vqe.feedback import FeedbackFormatter, FeedbackFormatType


def test_feedback_formatter_aggregated(
    mock_generated_results: base_schema.GeneratedResults,
    trial_result_df: pd.DataFrame,
) -> None:
    formatter = FeedbackFormatter(
        format_type=FeedbackFormatType.AGGREGATED_TEXT,
        generated_results=mock_generated_results,
        trial_result_df=trial_result_df,
    )
    output = formatter.format()
    assert "Idea Number: 1" in output
    assert "E1" in output
    assert "F1" in output
    assert "Cost Value: 0.1" in output
    assert "Each Molecule" not in output


def test_feedback_formatter_raw(
    mock_generated_results: base_schema.GeneratedResults,
    trial_result_df: pd.DataFrame,
    raw_result_df: pd.DataFrame,
) -> None:
    formatter = FeedbackFormatter(
        format_type=FeedbackFormatType.RAW_TEXT,
        generated_results=mock_generated_results,
        trial_result_df=trial_result_df,
        raw_result_df=raw_result_df,
    )
    output = formatter.format()
    assert "Idea Number: 1" in output
    assert "Molecule: H2" in output
    assert "Bond Length: 0.74" in output
    assert "Absolute Difference from Full CI: 0.001" in output


def test_feedback_formatter_no_raw_df(
    mock_generated_results: base_schema.GeneratedResults,
    trial_result_df: pd.DataFrame,
) -> None:
    formatter = FeedbackFormatter(
        format_type=FeedbackFormatType.RAW_TEXT,
        generated_results=mock_generated_results,
        trial_result_df=trial_result_df,
        raw_result_df=None,
    )
    with pytest.raises(ValueError, match="raw_result_df is not provided"):
        formatter.format()
