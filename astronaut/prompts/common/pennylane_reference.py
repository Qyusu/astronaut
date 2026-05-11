# flake8: noqa

from textwrap import dedent


class RelatedReferencePrompt:
    def __init__(self, reference: str, idea: str) -> None:
        developer_template = dedent(
            """
            # Role
            You are a quantum software engineer specializing in the PennyLane library.

            # Task
            Your task is to select, from the "# PennyLane Methods" list, the names of the PennyLane methods required to implement the ideas provided by the user. When making your selection, aim to maximize *recall*, not precision. If you are uncertain about whether a method is relevant, include it in the list.

            ## IMPORTANT CONSTRAINT
            **CRITICAL**: You MUST ONLY select method names that are explicitly listed in the "# PennyLane Methods" section below. Do NOT include any method names that are not present in this list, even if they exist in PennyLane or seem relevant. Only extract methods that are literally provided in the reference list.

            # PennyLane Methods
            Methods are provided in the format "{{method_name}}: {{description}}".
            ---
            {reference}
            ---

            ## Output Format
            - Output your selection as a JSON array of method names (each as a plain string), in the order provided in the # PennyLane Methods list.
            - If no methods match, return an empty list: [].
            - If a method fits multiple ideas, only include it once in the output list.

            Example:
            ["qml.RX", "qml.SingleExcitation"]
            """
        )

        self.developer_prompt = {
            "role": "developer",
            "content": developer_template.format(reference=reference),
        }

        user_template = dedent(
            """
            Please select the names of the PennyLane methods required to implement the below idea.

            {idea}
            """
        )
        self.user_prompt = {
            "role": "user",
            "content": user_template.format(idea=idea),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        return self.developer_prompt, self.user_prompt
