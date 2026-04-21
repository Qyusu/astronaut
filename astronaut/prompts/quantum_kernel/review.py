# flake8: noqa

from textwrap import dedent

from astronaut.prompts.quantum_kernel.constraints import (
    FEATURE_MAP_CONSTRAINTS,
    INPUT_DATA_CONSTRAINTS,
    PERFORMANCE_METRIC_DEFINITION,
)


class ReviewIdeaPrompt:
    def __init__(
        self,
        feature_map_constraints: str = FEATURE_MAP_CONSTRAINTS.strip(),
        performance_metric_definition: str = PERFORMANCE_METRIC_DEFINITION.strip(),
        input_data_constraints: str = INPUT_DATA_CONSTRAINTS.strip(),
        max_suggestion_num: int = 3,
    ) -> None:
        self.feature_map_constraints = feature_map_constraints
        self.performance_metric_definition = performance_metric_definition
        self.input_data_constraints = input_data_constraints
        self.max_suggestion_num = max_suggestion_num

        self.developer_template = dedent(
            """
            # Role
            You are an expert in quantum physics and quantum machine learning, specializing in quantum feature map design.


            # Task
            Your task is to review past ideas on quantum feature maps and their evaluation results, and propose improvements to enhance performance metric through continuous refinement. Restrict your review to the quantum feature map design itself; do not propose changes to the overall model, evaluation metrics, or other workflows.

            Encourage innovative approaches that go beyond simple combinations of established methods when suggesting improvements.


            # Feature Map Constraints
            {feature_map_constraints}

            # Performance Metric Definition
            {performance_metric_definition}

            # Input Data Constraints
            {input_data_constraints}


            # Output Format
            1. Keep Points:
                - Identify factors that contributed to improved performance metric.
                - Analyze the most accurate idea in detail.
                - Review multiple ideas to identify common elements that contributed to improving performance metric.
            2. Suggestions:
                - Limit the number of suggestions to {max_suggestion_num} or fewer.
                - Ensure each suggestion includes only a single proposal.
                - Each suggestion should be independent and not overlap with other suggestions.
                - Prioritize the most impactful suggestions.
                - If no suggestions for improvement are identified, return suggestions: ["COMPLETED"].
            3. Output Schema: {{
                "keep_points": ["point 1", "point 2", ..., "point n"],
                "suggestions": ["suggestion 1", "suggestion 2", ..., "suggestion n"]
                }}
            """
        )

    def _build_user_pormpt(
        self,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: str | None,
    ) -> dict[str, str]:
        template = dedent(
            """
            The previous idea and experimental results are provided below in the "# Previous Trial Idea and Results" section. These reflect iterative adjustments based on past trial review comments.

            Review the trial results to design a quantum feature map for a more accurate QSVM. Identify the factors that contributed to performance improvement and areas for further enhancement. Finally, check whether the review results comply with the design rules for the quantum feature map.

            # Previous Trial Idea and Results (Trial Number: {last_trial_num})
            {last_trial_results}
            """
        )

        if performance_review is not None:
            template = performance_review + template

        return {
            "role": "user",
            "content": template.format(
                last_trial_num=last_trial_num,
                last_trial_results=last_trial_results.strip(),
            ),
        }

    def build(
        self,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: str | None = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        developer_content = self.developer_template.format(
            feature_map_constraints=self.feature_map_constraints,
            performance_metric_definition=self.performance_metric_definition,
            input_data_constraints=self.input_data_constraints,
            max_suggestion_num=self.max_suggestion_num,
        )
        developer_prompt = {"role": "developer", "content": developer_content}
        user_prompt = self._build_user_pormpt(
            last_trial_num, last_trial_results, performance_review
        )

        return developer_prompt, user_prompt
