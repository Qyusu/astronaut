from textwrap import dedent

import pandas as pd

import astronaut.logics.quantum_kernel.constants as qk_const
import astronaut.logics.quantum_kernel.schema as qk_schema
import astronaut.logics.quantum_kernel.utils as qk_utils
import astronaut.schema as base_schema


class FeedbackFormatter:
    def __init__(
        self,
        generated_results: base_schema.GeneratedResults,
        trial_result_df: pd.DataFrame,
        raw_result_df: pd.DataFrame,
    ) -> None:
        """
        generated_results: GeneratedResults dataclass instance that contains idea, score, and implement.
        trial_result_df: "idea_id", "aggregated_run_time", "aggregated_{metric}_validation", "aggregated_{metric}_test"
        raw_result_df: raw single idea experiment result that contains each dataset result.
        """
        self.generated_results = generated_results
        self.trial_result_df = trial_result_df
        self.raw_result_df = raw_result_df

    def format(self) -> str:
        primary_metric, is_minimize = qk_utils.get_primary_metric_info()
        template = self._get_common_template()

        idea_order = []
        past_traials = []
        generated_result_list = self.generated_results.results
        sorted_result_df = self.trial_result_df.sort_values(
            f"{qk_const.AGGREGATED_COL_PREFIX}_{primary_metric}_{qk_const.VALIDATION_SUFFIX}",
            ascending=is_minimize,
        )

        for row in sorted_result_df.iterrows():
            idea_id = str(row[1][qk_const.IDEA_ID_COL])
            idea_num = int(idea_id.split("_")[-1])
            idea_order.append(f'"Idea Number: {idea_num}"')

            current_result = generated_result_list[idea_num - 1]

            format_kwargs = qk_schema.FeedbackTemplateVariables(
                idea_num=idea_num,
                idea_explanation=current_result.idea.explanation,
                idea_formula=current_result.idea.formula,
                idea_implementation=current_result.implement.code,
                run_time=row[1][qk_const.AGGREGATED_RUN_TIME_COL],
                aggregated_metric_label=f"Aggregated {primary_metric}",
                aggregated_metric_value=row[1][
                    f"{qk_const.AGGREGATED_COL_PREFIX}_{primary_metric}_{qk_const.VALIDATION_SUFFIX}"
                ],
                raw_result_str=self._get_raw_result_str(idea_id),
            ).model_dump()

            past_traial = template.format(**format_kwargs)
            past_traials.append(past_traial)

        direction_str = (
            "lowest to highest" if is_minimize else "highest to lowest"
        )
        idea_order_str = (
            f"Current trial ideas sorted by {primary_metric} from {direction_str}: "
            + " > ".join(idea_order)
        )
        past_traials_str = "\n".join(past_traials)

        return f"{idea_order_str}\n{past_traials_str}"

    def _get_common_template(self) -> str:
        return dedent(
            """
            ## Idea Number: {idea_num}
            ### Idea Explanation:
            {idea_explanation}

            ### Idea Mathematical Expression:
            {idea_formula}

            ### Idea Implementation:
            ```python
            {idea_implementation}
            ```

            ### Trial Results:
            #### Run Time: {run_time:.2f} [seconds]

            #### Result:
            - {aggregated_metric_label}: {aggregated_metric_value}

            #### Each Dataset Results:
            {raw_result_str}
            """
        )

    def _get_raw_result_str(self, idea_id: str) -> str:
        metrics = qk_utils.get_default_metrics_from_config()

        template_header = "\n- Dataset: {dataset_name}"
        metric_line_template = "\n    - {metric_label}: {metric_value}"

        raw_result_str = ""
        for _, row in self.raw_result_df[
            self.raw_result_df[qk_const.IDEA_ID_COL] == idea_id
        ].iterrows():
            raw_result_str += template_header.format(
                dataset_name=row[qk_const.DATASET_NAME_COL]
            )

            for metric in metrics:
                raw_result_str += metric_line_template.format(
                    metric_label=metric,
                    metric_value=row.get(metric, "N/A"),
                )

            raw_result_str += "\n"

        return raw_result_str.lstrip()
