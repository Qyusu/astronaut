from astronaut.prompts.common.expert import AskExpertPrompt


class TestAskExpertPrompt:
    def test_ask_expert_prompt(self) -> None:
        prompt = AskExpertPrompt(
            expert_role="Physicist", question="Why?", reference="Ref"
        )
        developer_prompt, user_prompt = prompt.build()
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "# Role\nYou are Physicist" in developer_prompt["content"]
        assert "## Question\nWhy?" in user_prompt["content"]
        assert "## Reference Information\nRef" in user_prompt["content"]
