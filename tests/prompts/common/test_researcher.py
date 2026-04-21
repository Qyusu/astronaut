from astronaut.prompts.common.researcher import ResearcherDeveloperPrompt


class TestResearcherDeveloperPrompt:
    def test_researcher_developer_prompt(self) -> None:
        prompt = ResearcherDeveloperPrompt()
        deveroper_prompt = prompt.get_prompt()
        assert deveroper_prompt["role"] == "developer"
        assert "# Role" in deveroper_prompt["content"]
