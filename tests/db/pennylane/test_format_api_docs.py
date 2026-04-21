import json
from pathlib import Path

from pytest_mock import MockerFixture

from astronaut.db.pennylane.format_api_docs import (
    build_prompt,
    format_docs_by_llm,
    format_method_info,
    load_raw_api_docs,
    save_formatted_api_docs,
)


def test_format_method_info() -> None:
    info = {
        "signature": "(a, b)",
        "docstring": "docs",
        "source": "def func(): pass",
    }
    result = format_method_info("func", info)
    assert "Method: func" in result
    assert "Signature: (a, b)" in result
    assert "Docstring: docs" in result
    assert "Source: def func(): pass" in result


def test_build_prompt() -> None:
    formatted_info = "Method info..."
    dev_prompt, user_prompt = build_prompt(formatted_info)

    assert dev_prompt["role"] == "developer"
    assert user_prompt["role"] == "user"
    assert formatted_info in user_prompt["content"]


def test_load_save_api_docs(tmp_path: Path) -> None:
    data = {"key": "value"}
    file_path = tmp_path / "test.json"

    with open(file_path, "w") as f:
        json.dump(data, f)

    loaded = load_raw_api_docs(file_path)
    assert loaded == data

    output_path = tmp_path / "output.json"
    save_formatted_api_docs(data, output_path)

    with open(output_path, "r") as f:
        saved = json.load(f)
    assert saved == data


def test_format_docs_by_llm(mocker: MockerFixture, tmp_path: Path) -> None:
    mock_chat_client_class = mocker.patch(
        "astronaut.db.pennylane.format_api_docs.ChatClient"
    )
    mock_chat_client = mock_chat_client_class.return_value

    # Mock LLM response
    mock_response = mocker.Mock()
    mock_response.content = json.dumps(
        {
            "args": [
                {
                    "name": "arg1",
                    "type": "int",
                    "required": True,
                    "description": "desc",
                }
            ],
            "description": "method desc",
        }
    )
    mock_response.cost = 0.01
    mock_chat_client.parse_chat.return_value = mock_response

    # Mock file operations
    raw_path = tmp_path / "raw.json"
    formatted_path = tmp_path / "formatted.json"

    with open(raw_path, "w") as f:
        json.dump(
            {"method1": {"signature": "()", "docstring": "", "source": ""}}, f
        )

    format_docs_by_llm(raw_path, formatted_path)

    assert formatted_path.exists()
    with open(formatted_path, "r") as f:
        result = json.load(f)

    assert "method1" in result
    assert result["method1"]["description"] == "method desc"
    assert result["method1"]["args"][0]["name"] == "arg1"
