# flake8: noqa

from textwrap import dedent


class SummaryPaperPrompt:
    def __init__(
        self,
        raw_content: str,
        max_summary_words: int = 1000,
    ) -> None:
        developer_template = dedent(
            """
            # Role
            You are an academic journal editor.

            # Task
            Your task is to thoroughly understand the full content of the paper provided by the user and summarize it. When summarizing, rely solely on the information from the provided paper and avoid referencing external sources. Ensure that the summary accurately reflects the authors' arguments and claims. Write a detailed summary of approximately {max_summary_words} words.

            # Output format
            - title: The title of the paper
            - summary: The detailed summary of the paper (approximately {max_summary_words} words)
            """
        )

        self.developer_prompt = {
            "role": "developer",
            "content": developer_template.format(
                max_summary_words=max_summary_words
            ),
        }

        user_template = dedent(
            """
            Follow these steps to create a summary of the paper:
            1. The full text of the paper is provided in the section titled "## Full content of paper" Carefully read and understand its content.
            2. Create a detailed summary with approximately {max_summary_words} words, focusing on the following aspects:
                - Key findings
                - Methodology
                - Results
                - Future works or potential areas for improvement

            ## Full content of paper
            {raw_content}
            """
        )
        self.user_prompt = {
            "role": "user",
            "content": user_template.format(
                raw_content=raw_content, max_summary_words=max_summary_words
            ),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        return self.developer_prompt, self.user_prompt
