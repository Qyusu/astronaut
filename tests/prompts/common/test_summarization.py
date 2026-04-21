from astronaut.prompts.common.summarization import SummaryPaperPrompt


class TestSummaryPaperPrompt:
    def test_summary_paper_prompt(self) -> None:
        prompt = SummaryPaperPrompt(
            raw_content="This is a paper.", max_summary_words=1000
        )
        developer_prompt, user_prompt = prompt.build()
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert (
            "summary: The detailed summary of the paper (approximately 1000 words)"
            in developer_prompt["content"]
        )
        assert (
            "2. Create a detailed summary with approximately 1000 words,"
            in user_prompt["content"]
        )
        assert (
            "## Full content of paper\nThis is a paper."
            in user_prompt["content"]
        )
