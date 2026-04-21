from typing import Any

import pytest

from astronaut.logics.common.generation import GenerateCode, GenerateIdea
from astronaut.schema import (
    GeneratedIdea,
    GeneratedIdeaResult,
    GeneratedImpl,
    GeneratedImplResult,
)


@pytest.fixture
def mock_client(mocker: Any) -> Any:
    return mocker.MagicMock()


@pytest.fixture
def generate_idea(mock_client: Any) -> GenerateIdea:
    return GenerateIdea(client=mock_client, model_version="test-model")


class TestGenerateIdea:
    def test_generate_success(
        self, generate_idea: GenerateIdea, mock_client: Any
    ) -> None:
        mock_response = mock_client.parse_chat.return_value
        mock_response.content = '{"results": [{"name": "Idea 1", "summary": "Sum", "explanation": "Expl", "formula": "Form", "key_sentences": ["Key"]}]}'
        mock_response.message_history = ["msg1"]
        mock_response.cost = 0.5

        dev_prompt = {"role": "system", "content": "dev"}
        user_prompt = {"role": "user", "content": "user"}

        result, history, cost = generate_idea.generate(
            developer_prompt=dev_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=None,
        )

        assert isinstance(result, GeneratedIdeaResult)
        assert len(result.results) == 1
        assert result.results[0].name == "Idea 1"
        assert history == ["msg1"]
        assert cost == 0.5
        assert mock_client.parse_chat.called


class TestGenerateCode:
    @pytest.fixture
    def generate_code(self, mock_client: Any, mocker: Any) -> GenerateCode:
        mocker.patch("astronaut.logics.common.generation.PennyLaneReference")
        return GenerateCode(
            client=mock_client,
            code_model_version="code-model",
            reference_model_version="ref-model",
        )

    def test_generate_success(
        self, generate_code: GenerateCode, mock_client: Any
    ) -> None:
        mock_response = mock_client.parse_chat.return_value
        mock_response.content = (
            '{"class_name": "MyFeatureMap", "code": "print(1)"}'
        )
        mock_response.message_history = ["code_msg"]
        mock_response.cost = 0.1

        dev_prompt = {"role": "system", "content": "dev"}
        user_prompt = {"role": "user", "content": "user"}

        result, history, cost = generate_code.generate(
            developer_prompt=dev_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=None,
        )

        assert isinstance(result, GeneratedImpl)
        assert result.class_name == "MyFeatureMap"
        assert result.code == "print(1)"
        assert cost == 0.1

    def test_generate_codes_batch(
        self, generate_code: GenerateCode, mocker: Any
    ) -> None:
        mocker.patch.object(generate_code, "generate")
        generate_code.generate.return_value = (
            GeneratedImpl(class_name="Impl", code="code"),
            ["msg"],
            0.1,
        )

        generate_code.pennylane_reference.extract_related_reference_str.return_value = "ref"

        idea_result = GeneratedIdeaResult(
            results=[
                GeneratedIdea(
                    name="Idea 1",
                    summary="S",
                    explanation="E",
                    formula="F",
                    key_sentences=["K"],
                )
            ]
        )

        prompt_builder = mocker.MagicMock()
        prompt_builder.return_value = (
            {"role": "system", "content": "dev"},
            {"role": "user", "content": "user"},
        )

        result, history, cost = generate_code.generate_codes(
            prompt_builder_fn=prompt_builder,
            last_code="old code",
            finalized_idea=idea_result,
            message_history=[],
        )

        assert isinstance(result, GeneratedImplResult)
        assert len(result.results) == 1
        assert cost == 0.1
        assert generate_code.generate.called

    def test_generate_codes_retry(
        self, generate_code: GenerateCode, mocker: Any
    ) -> None:
        mocker.patch.object(generate_code, "generate")
        generate_code.generate.side_effect = [
            Exception("First fail"),
            (GeneratedImpl(class_name="Impl", code="code"), ["msg"], 0.1),
        ]

        generate_code.pennylane_reference.extract_related_reference_str.return_value = "ref"
        mocker.patch("time.sleep")

        idea_result = GeneratedIdeaResult(
            results=[
                GeneratedIdea(
                    name="Idea 1",
                    summary="S",
                    explanation="E",
                    formula="F",
                    key_sentences=["K"],
                )
            ]
        )

        prompt_builder = mocker.MagicMock()
        prompt_builder.return_value = (
            {"role": "system", "content": "dev"},
            {"role": "user", "content": "user"},
        )

        result, history, cost = generate_code.generate_codes(
            prompt_builder_fn=prompt_builder,
            last_code="old code",
            finalized_idea=idea_result,
            message_history=[],
        )

        assert len(result.results) == 1
        assert generate_code.generate.call_count == 2
