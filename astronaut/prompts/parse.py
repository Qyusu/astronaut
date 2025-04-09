# flake8: noqa

from textwrap import dedent


class ParseJsonPrompt:
    def __init__(self, raw_content: str) -> None:
        self.system_prompt = {
            "role": "system",
            "content": dedent(
                """
                You are a helpful assistant capable of parsing strings into JSON-compatible formats.
                When given an input string, ensure it is transformed into a format that can be successfully parsed by Python's `json.loads`.
                If the input string is already in a JSON-compatible format, return it without modification.
                Handle edge cases like improper quotes, missing brackets, or incorrect separators.
                Always prioritize making the string valid JSON.
                """
            ),
        }

        template = dedent(
            """
            The following string needs to be parsed into a JSON-compatible format for use with Python's `json.loads`.
            If it is invalid or improperly formatted, correct it. 

            The value corresponding to each JSON key is enclosed with the same tag name as the key. Please perform formatting only without altering the meaning of the text.
        
            Here is the input:
            "{raw_content}"
            """
        )
        self.user_prompt = {
            "role": "user",
            "content": template.format(raw_content=raw_content),
        }

    def build(self) -> tuple[dict[str, str], dict[str, str]]:
        return self.system_prompt, self.user_prompt
