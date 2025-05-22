# flake8: noqa

from textwrap import dedent

from astronaut.constants import NOT_PROVIDED_INFORMATION
from astronaut.prompts.quantum_kernel.scoring_few_shots import SCORING_FEW_SHOTS


class ScoringIdeaPrompt:
    def __init__(self, score_histories: str = "") -> None:
        self.system_prompt = {
            "role": "system",
            "content": dedent(
                """
                As a reviewer for a scientific journal, you are tasked with evaluating new scientific ideas from multiple perspectives while adhering to specific evaluation criteria. Your evaluation should be thorough, objective, and based on the guidelines provided below.

                # Evaluation Criteria
                Each criterion is scored on a scale of 0.0 to 10.0, in increments of 0.1, where 0.0 represents the lowest score and 10.0 represents the highest score:
                - Originality: Assess how the idea differs from existing research. Does it make a novel contribution?
                - Feasibility: Evaluate the practicality of implementing the idea.
                - Versatility: Consider how broadly the idea can be applied.

                # Steps for Evaluation
                1. Understand the Idea: 
                    - Carefully read and comprehend the proposed idea.
                    - Organize the information needed to make a well-informed evaluation.
                2. Assess Information Sufficiency: 
                    - First, confirm the round number provided by the user. If it is the final round, skip Step 2 and proceed to Step 3.
                    - Otherwise, assess whether the "# Related Work" section provides sufficient information to evaluate the idea.
                    - If the related work information is "{not_provided_information}", skip Step 2 and proceed to Step 3.
                    - If the information is insufficient for scoring, <is_lack_information> tag set to True, and list up to 5 necessary information key sentences as a comma-separated list within <additional_key_sentences> tags. The search will be conducted by embedded vector for academic paper; therefore, ensure the <additional_key_sentences> are specific and relevant. Each sentence length should be between 50 to 100 words. In this case, terminate the scoring process and set all scores to 0.0.
                    - If the information is sufficient, proceed to Step 3.
                3. Provide Reasoning:
                    - Proceed to evaluate the idea based on the specified criteria.
                    - If the related work information is "{not_provided_information}", use your own knowledge to evaluate the idea. DO NOT request additional information and <is_lack_information> tag set to False.
                    - Enclose the rationale behind the evaluation results of each indicator in <reason> tags and explain it in text.
                4. Assign Scores:
                    - Based on the evaluation results and their rationale, assign a score to each indicator.
                5. Terminating the Evaluation:
                    - Once all scores and reasoning have been assigned, the evaluation is complete. <is_lack_information> tag set to False.

                # Baseline
                {few_shot_examples}
                """.format(
                    not_provided_information=NOT_PROVIDED_INFORMATION,
                    few_shot_examples=SCORING_FEW_SHOTS,
                )
                + score_histories
            ),
        }

    def build(self, idea: str, related_work: str, round: int, max_round: int) -> tuple[dict[str, str], dict[str, str]]:
        if round == 1:
            # First round prompt
            template = dedent(
                """
                Round {current_round}/{max_scoring_round}.

                You are tasked with evaluating the following "# Proposed Idea" using the specified criteria and providing scores for each criterion. The idea represents a newly proposed quantum feature map for the quantum kernel method. Your evaluation and scoring should consider multiple perspectives and adhere to the strictest possible standards. Finally, after all the scores have been assigned, summarize the rationale for each score.

                # Proposed Idea
                {idea}

                # Related Work
                {related_work}
                """
            )
            user_prompt = {
                "role": "user",
                "content": template.format(
                    current_round=round, max_scoring_round=max_round, idea=idea, related_work=related_work
                ),
            }
        elif round == max_round:
            # Final round prompt
            tempalte = dedent(
                """
                Round {current_round}/{max_scoring_round} (Final Round).

                Additional Related Work has been provided for further context.

                This marks the final opportunity to request additional information. Based on the provided details, conduct a comprehensive evaluation and ensure scores are assigned to each criterion without exception.
                This round DOES NOT return "is_lack_information"=True. You must provide scores for each criterion.

                # Related Work
                {related_work}
                """
            )
            user_prompt = {
                "role": "user",
                "content": tempalte.format(current_round=round, max_scoring_round=max_round, related_work=related_work),
            }
        else:
            # Intermediate round prompt
            tempalte = dedent(
                """
                Round {current_round}/{max_scoring_round}.
                
                Additional Related Work has been provided for further context. Please evaluate the proposed idea using the specified criteria and provide scores for each criterion.

                # Related Work
                {related_work}
                """
            )
            user_prompt = {
                "role": "user",
                "content": tempalte.format(current_round=round, max_scoring_round=max_round, related_work=related_work),
            }
        return self.system_prompt, user_prompt
