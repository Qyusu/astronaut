from enum import Enum, auto
from textwrap import dedent

import pandas as pd

from astronaut.schema import GeneratedResults


class FeedbackFormatType(Enum):
    AGGREGATED_TEXT = auto()
    RAW_TEXT = auto()
    TRAJECTORY_IMAGE = auto()


class FeedbackFormatter:
    def __init__(
        self,
        format_type: FeedbackFormatType,
        generated_results: GeneratedResults,
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
            self.raw_result_df["idea_id"] == idea_id
        ]
        molecule_bond_length_result_df = (
            molecule_bond_length_result_df.sort_values(
                ["molecule_name", "bond_length"], ascending=True
            )
        )
        molecule_bond_length_results_str = ""
        for (
            molecule_name,
            molecule_group,
        ) in molecule_bond_length_result_df.groupby("molecule_name"):
            molecule_bond_length_results_str += f"- Molecule: {molecule_name}\n"
            for bond_length_row in molecule_group.iterrows():
                bond_length = bond_length_row[1]["bond_length"]
                cost = bond_length_row[1]["cost"]
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
            "aggregated_cost", ascending=True
        )

        for row in sorted_result_df.iterrows():
            idea_id = str(row[1]["idea_id"])
            idea_num = int(idea_id.split("_")[-1])
            idea_order.append(f'"Idea Number: {idea_num}"')

            current_result = generated_result_list[idea_num - 1]

            format_kwargs = {
                "idea_num": idea_num,
                "idea_explanation": current_result.idea.explanation,
                "idea_formula": current_result.idea.formula,
                "num_gates_order": row[1]["num_gates_order"],
                "num_params_order": row[1]["num_params_order"],
                "run_time": row[1]["aggregated_run_time"],
                "num_iter": row[1]["aggregated_num_iter"],
                "cost": row[1]["aggregated_cost"],
            }

            if include_molecule_details:
                format_kwargs["molecule_bond_length_results_str"] = (
                    self._get_molecule_bond_length_results(idea_id)
                )

            past_traial = template.format(**format_kwargs)
            past_traials.append(past_traial)

        idea_order_str = (
            "Current trial ideas sorted by cost (cost from lowest to highest): "
            + " < ".join(idea_order)
        )
        past_traials_str = "\n".join(past_traials)

        return f"{idea_order_str}\n{past_traials_str}"

    def _format_image_by_trajectory(self) -> str:
        # [TODO]: 各Bond lengthの最適化の軌跡を画像として入力
        raise NotImplementedError
