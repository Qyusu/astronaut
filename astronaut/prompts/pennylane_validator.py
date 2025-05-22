# flake8: noqa

from textwrap import dedent


class PennyLaneDocsValidatePrompt:
    def __init__(self, methods: str, references: str) -> None:
        self.methods = methods
        self.references = references

        self.system_prompt = {
            "role": "system",
            "content": dedent(
                """
                You are an expert quantum software engineer especially skilled in the PennyLane library.

                Your Task is to extracted argments from the user provided PennyLane function and PennyLane documentation. The Details are as follows:
                # Task
                For each function or class in the provided input, follow these steps:

                1. **Extract the Class Name**:
                    - Identify the function or class name from the user-provided code in the "# User Provided PennyLane Class and Method" section.
                    - The class name starts with `qml.` and ends before the first `(`.
                    - Do not include the parentheses or any characters after them.
                2. **Extract User-Defined Argument Names**:
                    - Identify argument names in the user code based on the patterns described in the "# Expected User Arguments Pattern" section.
                    - For **Pattern 1**, arguments are values only (no `arg_name` or `=`). This pattern is excluded from validation and should be ignored in this step.
                    - For **Pattern 2** and **Pattern 3**, extract the `arg_name` portion only. If multiple arguments are present, separate them by commas and exclude type hints, values, and assignment operators (`=`).
                3. **Extract Reference Argument Names from Documentation**:
                    - Refer to the corresponding "Class Name" section in the "# PennyLane Documentation" part.
                    - The arguments are listed as `"name" ("type"): "description"`. Only extract the `name` portion.
                    - If multiple arguments exist, separate them by commas.

                # Expected User Arguments Pattern
                - Pattern 1: 
                    - qml.Hoge(`value`) => Expected: ignore
                    - qml.Hoge(`value_1`, `value_2`) => Expected: ignore
                - Pattern 2: 
                    - qml.Hoge(`arg_name = value`) => Expected: arg_name
                    - qml.Hoge(`arg_name=value`) => Expected: arg_name
                    - qml.Hoge(`arg_name_1=value_1`, `arg_name_2=value_2`) => Expected: arg_name_1, arg_name_2
                    - qml.Hoge(`value_1`, `arg_name_2=value_2`) => Expected: arg_name_2
                - Pattern 3:
                    - qml.Hoge(`arg_name: arg_type = value`) => Expected: arg_name
                    - qml.Hoge(`arg_name:arg_type=value`) => Expected: arg_name
                    - qml.Hoge(`arg_name_1:arg_type=value_1`, `arg_name_2:arg_type=value_2`) => Expected: arg_name_1, arg_name_2
                    - qml.Hoge(`value_1`, `arg_name_2:arg_type=value_2`) => Expected: arg_name_2

                # Output JSON Format
                ``` json
                [
                    {{
                        "class_name": "Name of the user provided PennyLane class (step1)",
                        "user_args_name": "Argment name list that extracted user code (step2)",
                        "docs_args_name": "Argument name list that extracted pennylane documentation (step3)"
                    }},
                    // Repeat for each case
                ]
                ```
                """
            ),
        }
        template = dedent(
            """
            Pennylane method and class list is provided in "# User Provided PennyLane Class and Method" section. PennyLane documentation is provided in "# PennyLane Documentation" section. Plese extract the argments from the user provided PennyLane method and PennyLane documentation.

            Make sure the output format is defined in "# Output JSON Format" section.

            # User Provided PennyLane Class and Method
            {methods}

            # PennyLane Documentation
            {references}
            """
        )
        self.user_prompt = {
            "role": "user",
            "content": template.format(methods=methods, references=self.references),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        return self.system_prompt, self.user_prompt
