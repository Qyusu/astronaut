from astronaut.prompts.common.parse import ParseJsonPrompt


class TestParseJsonPrompt:
    def test_parse_json_prompt(self) -> None:
        prompt = ParseJsonPrompt(raw_content="{'a': 1}")
        developer_prompt, user_prompt = prompt.build()
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "Here is the input:\n\"{'a': 1}\"" in user_prompt["content"]
