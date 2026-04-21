from textwrap import dedent

import astronaut.constants as exp_const


class FixCodePrompt:
    def __init__(
        self,
        code: str,
        error_messages: list[str],
        warning_messages: list[str],
        pennylane_version: str = exp_const.PENNYLANE_VERSION,
    ) -> None:
        self.code = code
        self.error_messages_string = (
            "- " + "\n- ".join(error_messages)
            if len(error_messages) > 0
            else "Errors not found."
        )
        self.warning_messages_string = (
            "- " + "\n- ".join(warning_messages)
            if len(warning_messages) > 0
            else "Warnings not found."
        )
        self.pennylane_version = pennylane_version

        self.developer_prompt = {
            "role": "developer",
            "content": dedent(
                """
                # Role
                You are an expert quantum computing software engineer using the PennyLane library in Python.

                # Task
                Your task is to fix the errors and warnings in the code provided by the user. Only fix errors and warnings—do not change the logic or structure of the code.

                # Output Format
                Make sure the code follows the same structure as the base code provided and is formatted in the following JSON format:
                    - class_name: The name of the generated class
                    - code: The generated code
                """
            ),
        }

        tempalte = dedent(
            """
            Please fix the following errors and warnings in the code. The PennyLane library used in the code is version {pennylane_version}.

            ## Code to Fix
            ```python
            {code}
            ```

            ## Errors
            {error_messages_string}

            ## Warnings
            {warning_messages_string}
            """
        )
        self.user_prompt = {
            "role": "user",
            "content": tempalte.format(
                code=self.code,
                error_messages_string=self.error_messages_string,
                warning_messages_string=self.warning_messages_string,
                pennylane_version=self.pennylane_version,
            ),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        return self.developer_prompt, self.user_prompt
