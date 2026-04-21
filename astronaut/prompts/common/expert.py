# flake8: noqa

from textwrap import dedent


class AskExpertPrompt:
    def __init__(self, expert_role: str, question: str, reference: str) -> None:
        developer_template = dedent(
            """
            # Role
            You are {expert_role}.

            # Task
            The user is asking you a question expecting your expertise and specialized knowledge. Please answer the question based on the reference information provided by the user, utilizing your professional expertise.
            Your response should be as specific and clear as possible with no particular word limit. If the question is outside your area of expertise, please respond with "Your question is outside my area of expertise" instead of attempting to answer.

            # Output format
            The response should be structured as follows. The response parsed by Python's json.loads() should be a valid JSON object.
            - expert_role: The role of the expert
            - question: The question that the user wants to ask the expert
            - answer: The answer of the expert
            """
        )

        self.developer_prompt = {
            "role": "developer",
            "content": developer_template.format(expert_role=expert_role),
        }

        user_template = dedent(
            """
            Please answer the question based on the reference information and your expertise.

            ## Question
            {question}

            ## Reference Information
            {reference}
            """
        )
        self.user_prompt = {
            "role": "user",
            "content": user_template.format(
                question=question, reference=reference
            ),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        return self.developer_prompt, self.user_prompt
