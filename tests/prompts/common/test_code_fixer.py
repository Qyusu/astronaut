from astronaut.prompts.common.code_fixer import FixCodePrompt


class TestFixCodePrompt:
    def test_fix_code_prompt(self) -> None:
        prompt = FixCodePrompt(
            code="import pennylane as qml",
            error_messages=["Error 1"],
            warning_messages=["Warning 1"],
            pennylane_version="0.38.0",
        )
        developer_prompt, user_prompt = prompt.build()
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert (
            "```python\nimport pennylane as qml\n```" in user_prompt["content"]
        )
        assert "## Errors\n- Error 1" in user_prompt["content"]
        assert "## Warnings\n- Warning 1" in user_prompt["content"]
        assert "the code is version 0.38.0." in user_prompt["content"]

    def test_fix_code_prompt_with_no_error_and_no_warning(self) -> None:
        prompt = FixCodePrompt(
            code="import pennylane as qml",
            error_messages=[],
            warning_messages=[],
            pennylane_version="0.38.0",
        )
        _, user_prompt = prompt.build()
        assert "## Errors\nErrors not found." in user_prompt["content"]
        assert "## Warnings\nWarnings not found." in user_prompt["content"]
