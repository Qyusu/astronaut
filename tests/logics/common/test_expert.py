from typing import Any

import pytest

from astronaut.logics.common.expert import PaperExpert
from astronaut.schema import ExpertResult


@pytest.fixture
def mock_clients(mocker: Any) -> tuple[Any, Any]:
    llm = mocker.MagicMock()
    db = mocker.MagicMock()
    return llm, db


@pytest.fixture
def paper_expert(mock_clients: tuple[Any, Any]) -> PaperExpert:
    llm, db = mock_clients
    return PaperExpert(
        llm_client=llm,
        paper_db_client=db,
        model_version="test-model",
        local_paper_dir="/tmp",
    )


class TestPaperExpert:
    def test_request_expert_response_success(
        self,
        paper_expert: PaperExpert,
        mock_clients: tuple[Any, Any],
        mocker: Any,
    ) -> None:
        llm, db = mock_clients

        db.query.return_value = {
            "matches": [
                {"metadata": {"document_id": "doc1"}},
                {"metadata": {"document_id": "doc2"}},
            ]
        }

        mocker.patch.object(
            paper_expert, "load_full_text_from_local", return_value="Paper text"
        )
        mocker.patch.object(
            paper_expert,
            "cut_string_if_over_token_limit",
            side_effect=lambda x, y: x,
        )

        mock_response = mocker.MagicMock()
        mock_response.content = (
            '{"expert_role": "Physicist", "question": "Q?", "answer": "Answer"}'
        )
        mock_response.cost = 0.01
        llm.parse_chat.return_value = mock_response

        result, cost = paper_expert.request_expert_response(
            expert_role="Physicist",
            question="Q?",
            top_k=2,
            max_paper_tokens=100,
        )

        assert isinstance(result, ExpertResult)
        assert result.answer == "Answer"
        assert cost == 0.01
        assert llm.parse_chat.called
        assert db.query.called

    def test_request_expert_response_llm_error(
        self,
        paper_expert: PaperExpert,
        mock_clients: tuple[Any, Any],
        mocker: Any,
    ) -> None:
        llm, db = mock_clients
        db.query.return_value = {"matches": []}
        llm.parse_chat.side_effect = Exception("LLM Error")

        result, cost = paper_expert.request_expert_response(
            expert_role="Physicist", question="Q?"
        )

        assert (
            result.answer
            == "Expert answer could not be acquired due to parsing error."
        )
        assert cost == 0.0
