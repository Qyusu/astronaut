# flake8: noqa

from textwrap import dedent
from typing import Optional

from astronaut.constants import REASONING_SERIES


class ReviewIdeaPrompt:
    def __init__(self, llm_model_version: str, max_suggestion_num: int = 3) -> None:
        self.llm_model_version = llm_model_version
        self.max_suggestion_num = max_suggestion_num

        self.developer_template = dedent(
            """
            You are an expert in quantum physics and quantum machine learning, specializing in quantum feature map design.

            # Task
            Your task is to review past ideas on quantum feature maps and their evaluation results, and propose improvements to enhance accuracy through continuous refinement. Restrict your review to the quantum feature map design itself; do not propose changes to the overall model, evaluation metrics, or other workflows.

            # Quantum Feature Map Definition
            Quantum feature maps (Φ(x)) will be used to compute the quantum kernel K(x, x') = |⟨Φ(x)|Φ(x')⟩|^2 for a QSVM (Quantum Support Vector Machine).
            1. **Design Considerations**:
                - Define combinations of quantum gates.
                - Define an entanglement pattern for the features.
                - Specify the method for embedding input data and quantum states as rotation angles of quantum gates.
                - Ensure the 80-dimensional input data is utilized effectively, minimizing any loss of information.
                    - Avoid excessive feature compression that may lead to information loss (e.g., simple feature averaging, summing, etc.).
            2. **Restrictions on Encoding and Embedding**:
                - Only **linear functions** are allowed for encoding and embedding.
                - All parameters in the encoding and embedding must be **non-trainable**.

            # Output Format
            1. Keep Points:
                - Identify factors that contributed to improved accuracy.
                - Analyze the most accurate idea in detail.
                - Review multiple ideas to identify common elements that contributed to improving accuracy.
            2. Suggestions:
                - Limit the number of suggestions to {max_suggestion_num} or fewer.
                - Ensure each suggestion includes only a single proposal.
                - Prioritize the most impactful suggestions.
                - If no suggestions for improvement are identified, return suggestions: ["COMPLETED"].
            3. Output Schema: {{
                "keep_points": ["point 1", "point 2", ..., "point n"],
                "suggestions": ["suggestion 1", "suggestion 2", ..., "suggestion n"]
                }}.

            # Notes
            - Input Data: The input data, originally represented as 784-dimensional image data, has been compressed to 80 dimensions using PCA and each value normalized to the range [0.0, 1.0]
            - Simulation: Use an ideal quantum simulator without noise for evaluation
            - Evaluation Metric: Classification accuracy is the primary metric. The goal is to achieve the highest possible accuracy.
            """
        )

    def _build_user_pormpt(
        self,
        last_trial_num: int,
        last_trial_results: str,
        performance_review: Optional[str],
    ) -> dict[str, str]:
        template = dedent(
            """
            The previous idea and experimental results are provided below in the "# Previous Trial Idea and Results" section. These reflect iterative adjustments based on past trial review comments. 
            
            Review the trial results to design a quantum feature map for a more accurate QSVM. Identify the factors that contributed to accuracy improvement and areas for further enhancement. Finally, check whether the review results comply with the design rules for the quantum feature map.

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
        performance_review: Optional[str] = None,
    ) -> tuple[dict[str, str], dict[str, str]]:
        developer_content = self.developer_template.format(max_suggestion_num=self.max_suggestion_num)
        if self.llm_model_version in REASONING_SERIES:
            system_prompt = {"role": "developer", "content": developer_content}
        else:
            system_prompt = {"role": "system", "content": developer_content}

        user_prompt = self._build_user_pormpt(last_trial_num, last_trial_results, performance_review)

        return system_prompt, user_prompt
