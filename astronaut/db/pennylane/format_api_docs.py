import json
from pathlib import Path
from textwrap import dedent

from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from astronaut.configs import settings
from astronaut.llm import ChatClient
from astronaut.llm.config import OpenAIConfig

OPENAI_MODEL_VERSION = "gpt-4.1-2025-04-14"


class Arg(BaseModel):
    name: str = Field(..., description="The name of the argument.")
    type: str = Field(..., description="The type of the argument.")
    required: bool = Field(..., description="Whether the argument is required.")
    description: str = Field(
        ..., description="The description of the argument."
    )


class APIDocResult(BaseModel):
    args: list[Arg] = Field(..., description="The arguments of the method.")
    description: str = Field(..., description="The description of the method.")


def build_prompt(
    formatted_method_info: str,
) -> tuple[dict[str, str], dict[str, str]]:
    developer_prompt_template = dedent(
        """
        You are a software engineer with excellent explanatory skills.
        You will be provided with the method name, its arguments and their types,
        the method's docstring, and the method's source code.

        Your task is to extract each argument's name and type;
        for each argument, generate a description of the values that should be supplied;
        and produce a detailed explanation of the method's functionality and intended usage.

        When creating all descriptions, use only the provided source code and docstring—prioritizing
        the source code whenever the docstring may be outdated or conflicting.
        """
    )

    user_prompt_template = """
            Please format the following method information into a structured format.

            # Method Information
            {formatted_method_info}

            # Output Format
            ```json
            {{
            "args": [
                {{
                "name": "argment name",
                "type": "argment type",
                "required": True,
                "description": "The description of the argument."
                }},
                {{
                "name": "argment name",
                "type": "argment type",
                "required": False,
                "description": "The description of the argument."
                }}
            ],
            "description": "detailed description of the method"
            }}
            ```
            """

    developer_prompt = {
        "role": "developer",
        "content": developer_prompt_template,
    }

    user_prompt = {
        "role": "user",
        "content": user_prompt_template.format(
            formatted_method_info=formatted_method_info
        ),
    }
    return developer_prompt, user_prompt


def format_method_info(method_name: str, method_info: dict) -> str:
    return f"""
    Method: {method_name}
    Signature: {method_info["signature"]}
    Docstring: {method_info["docstring"]}
    Source: {method_info["source"]}
    """


def load_raw_api_docs(raw_file_path: Path) -> dict:
    with open(raw_file_path, "r") as f:
        return json.load(f)


def save_formatted_api_docs(
    formatted_api_docs: dict, formatted_file_path: Path
) -> None:
    with open(formatted_file_path, "w") as f:
        json.dump(formatted_api_docs, f)


def format_docs_by_llm(raw_file_path: Path, formatted_file_path: Path) -> None:
    client = ChatClient(
        OpenAIConfig(
            api_key=settings.OPENAI_API_KEY,
            default_model_version=settings.DEFAULT_MODEL_VERSION,
        )
    )
    total_cost = 0.0

    results_dict = {}
    api_docs = load_raw_api_docs(raw_file_path)
    for i, (method_name, method_info) in enumerate(tqdm(api_docs.items())):
        formatted_method_info = format_method_info(method_name, method_info)
        developer_prompt, user_prompt = build_prompt(formatted_method_info)

        response = client.parse_chat(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=0,
            n=1,
            max_tokens=None,
            response_format=APIDocResult,
            model_version=OPENAI_MODEL_VERSION,
        )

        result = json.loads(response.content)
        results_dict[method_name] = result

        total_cost += response.cost

        if i % 30 == 0:
            save_formatted_api_docs(results_dict, formatted_file_path)
            logger.info(
                f"Saved {len(results_dict)} methods. Cost: ${total_cost}"
            )

    save_formatted_api_docs(results_dict, formatted_file_path)
    logger.info(
        f'All methods processed and saved to "{formatted_file_path}". Cost: ${total_cost}'
    )
