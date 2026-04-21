from __future__ import annotations

from enum import Enum, auto
from textwrap import dedent
from typing import TYPE_CHECKING

import pandas as pd

import astronaut.logics.vqe.constants as vqe_const
from astronaut.logics.vqe.schema import FeedbackTemplateVariables

if TYPE_CHECKING:
    from astronaut.schema import GeneratedResults


class FeedbackFormatType(Enum):
    AGGREGATED_TEXT = auto()
    RAW_TEXT = auto()
    TRAJECTORY_IMAGE = auto()


class FeedbackFormatter:
    def __init__(
        self,
        format_type: FeedbackFormatType,
        generated_results: "GeneratedResults",
        trial_result_df: pd.DataFrame,
        raw_result_df: pd.DataFrame | None = None,
    ) -> None:
        """
        generated_results: GeneratedResults dataclass instance that contains idea, score, and implement.
        trial_result_df: "idea_id", "num_gates_order", "num_params_order", "aggregated_num_iter", "aggregated_run_time", "aggregated_cost"
        raw_result_df: raw single idea experiment result that contains each molecule and bond length result.
        """
        self.format_type = format_type
        self.generated_results = generated_results
        self.trial_result_df = trial_result_df
        self.raw_result_df = raw_result_df

    def format(self) -> str:
        if self.format_type == FeedbackFormatType.AGGREGATED_TEXT:
            return self._format_text_by_aggregated_result()
        elif self.format_type == FeedbackFormatType.RAW_TEXT:
            return self._format_text_by_raw_result()
        elif self.format_type == FeedbackFormatType.TRAJECTORY_IMAGE:
            return self._format_image_by_trajectory()
        else:
            raise ValueError(f"Invalid format type: {self.format_type}")

    def _format_text_by_aggregated_result(self) -> str:
        return self._format_text_common(include_molecule_details=False)

    def _format_text_by_raw_result(self) -> str:
        if self.raw_result_df is None:
            raise ValueError(
                "raw_result_df is not provided. Please provide raw cost result of each molecule and each bond length."
            )
        return self._format_text_common(include_molecule_details=True)

    def _get_common_template(
        self, include_molecule_details: bool = False
    ) -> str:
        """Get the common template with optional molecule details section."""
        base_template = dedent(
            """
            ## Idea Number: {idea_num}
            ### Idea Explanation:
            {idea_explanation}

            ### Idea Mathematical Expression:
            {idea_formula}

            ### Ansatz Scalability:
            - Number of Gates Scaling Order: {num_gates_order}
            - Number of Parameters Scaling Order: {num_params_order}

            ### Trial Results:
            #### Number of Iterations (Mean): {num_iter:.2f}
            #### Run Time (Mean): {run_time:.2f} [seconds]

            #### Result:
            - Cost Value: {cost}
            """
        )

        if include_molecule_details:
            base_template += dedent(
                """
                #### Each Molecule and Bond Length Results:
                {molecule_bond_length_results_str}
                """
            )

        return base_template

    def _get_molecule_bond_length_results(self, idea_id: str) -> str:
        """Extract molecule and bond length results for a given idea_id."""
        if self.raw_result_df is None:
            return "Not provided molecule and bond length results"

        if self.raw_result_df.empty:
            return "All experiments in current trial are failed. Not provided molecule and bond length results"

        molecule_bond_length_result_df = self.raw_result_df[
            self.raw_result_df[vqe_const.IDEA_ID_COL] == idea_id
        ]
        molecule_bond_length_result_df = (
            molecule_bond_length_result_df.sort_values(
                [vqe_const.MOLECULE_NAME_COL, vqe_const.BOND_LENGTH_COL],
                ascending=True,
            )
        )
        molecule_bond_length_results_str = ""
        for (
            molecule_name,
            molecule_group,
        ) in molecule_bond_length_result_df.groupby(
            vqe_const.MOLECULE_NAME_COL
        ):
            molecule_bond_length_results_str += f"- Molecule: {molecule_name}\n"
            for bond_length_row in molecule_group.iterrows():
                bond_length = bond_length_row[1][vqe_const.BOND_LENGTH_COL]
                cost = bond_length_row[1][vqe_const.COST_COL]
                molecule_bond_length_results_str += f"  - Bond Length: {bond_length} [Å], Absolute Difference from Full CI: {cost} [Ha]\n"

        return (
            molecule_bond_length_results_str
            if molecule_bond_length_results_str != ""
            else "Not provided molecule and bond length results"
        )

    def _format_text_common(
        self, include_molecule_details: bool = False
    ) -> str:
        """Common formatting logic for both aggregated and raw result formats."""
        template = self._get_common_template(include_molecule_details)

        idea_order = []
        past_traials = []
        generated_result_list = self.generated_results.results
        sorted_result_df = self.trial_result_df.sort_values(
            vqe_const.AGGREGATED_COST_COL, ascending=True
        )

        for row in sorted_result_df.iterrows():
            idea_id = str(row[1][vqe_const.IDEA_ID_COL])
            idea_num = int(idea_id.split("_")[-1])
            idea_order.append(f'"Idea Number: {idea_num}"')

            current_result = generated_result_list[idea_num - 1]

            if include_molecule_details:
                molecule_bond_length_results_str = (
                    self._get_molecule_bond_length_results(idea_id)
                )
            else:
                molecule_bond_length_results_str = ""

            format_kwargs = FeedbackTemplateVariables(
                idea_num=idea_num,
                idea_explanation=current_result.idea.explanation,
                idea_formula=current_result.idea.formula,
                num_gates_order=row[1][vqe_const.NUM_GATES_ORDER_COL],
                num_params_order=row[1][vqe_const.NUM_PARAMS_ORDER_COL],
                run_time=row[1][vqe_const.AGGREGATED_RUN_TIME_COL],
                num_iter=row[1][vqe_const.AGGREGATED_NUM_ITER_COL],
                cost=row[1][vqe_const.AGGREGATED_COST_COL],
                molecule_bond_length_results_str=molecule_bond_length_results_str,
            ).model_dump()

            past_traial = template.format(**format_kwargs)
            past_traials.append(past_traial)

        idea_order_str = (
            "Current trial ideas sorted by cost from lowest to highest: "
            + " < ".join(idea_order)
        )
        past_traials_str = "\n".join(past_traials)

        return f"{idea_order_str}\n{past_traials_str}"
