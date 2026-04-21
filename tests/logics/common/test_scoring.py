from typing import Any

import pytest

from astronaut.logics.common.scoring import ScoringIdea
from astronaut.schema import GeneratedIdea


@pytest.fixture
def mock_clients(mocker: Any) -> tuple[Any, Any]:
    llm = mocker.MagicMock()
    db = mocker.MagicMock()
    return llm, db


@pytest.fixture
def scoring_idea(mock_clients: tuple[Any, Any]) -> ScoringIdea:
    llm, db = mock_clients
    return ScoringIdea(
        llm_client=llm, paper_db_client=db, model_version="test-model"
    )


@pytest.fixture
def idea() -> GeneratedIdea:
    return GeneratedIdea(
        name="Test",
        summary="Sum",
        explanation="Expl",
        formula="Form",
        key_sentences=["K1"],
    )


class TestScoringIdea:
    def test_no_scoring_process(
        self, scoring_idea: ScoringIdea, idea: GeneratedIdea
    ) -> None:
        result, cost = scoring_idea.scoring_with_related_works(
            idea, max_scoring_round=0
        )
        assert result.score.originality.score is None
        assert cost == 0.0

    def test_scoring_success(
        self,
        scoring_idea: ScoringIdea,
        idea: GeneratedIdea,
        mock_clients: tuple[Any, Any],
        mocker: Any,
    ) -> None:
        llm, db = mock_clients

        db.query.return_value = {
            "matches": [
                {
                    "id": "p1",
                    "metadata": {"abstract": "abs", "chunk_text": "chunk"},
                }
            ]
        }

        mock_response = mocker.MagicMock()
        mock_response.content = '{"score": {"originality": {"score": 0.9, "reason": "r"}, "feasibility": {"score": 0.7, "reason": "r"}, "versatility": {"score": 0.6, "reason": "r"}}, "is_lack_information": false, "additional_key_sentences": []}'
        mock_response.cost = 0.05
        llm.parse_chat.return_value = mock_response

        result, cost = scoring_idea.scoring_with_related_works(
            idea, max_scoring_round=1
        )

        assert result.score.originality.score == 0.9
        assert result.score.feasibility.score == 0.7
        assert result.score.versatility.score == 0.6
        assert cost == 0.05
        assert db.query.called

    def test_scoring_no_db(
        self, mock_clients: tuple[Any, Any], idea: GeneratedIdea, mocker: Any
    ) -> None:
        llm, _ = mock_clients
        scoring_no_db = ScoringIdea(
            llm_client=llm, paper_db_client=None, model_version="m"
        )

        mock_response = mocker.MagicMock()
        mock_response.content = '{"score": {"originality": {"score": 0.7, "reason": "r"}, "feasibility": {"score": 0.7, "reason": "r"}, "versatility": {"score": 0.6, "reason": "r"}}, "is_lack_information": false, "additional_key_sentences": []}'
        mock_response.cost = 0.01
        llm.parse_chat.return_value = mock_response

        result, cost = scoring_no_db.scoring_with_related_works(
            idea, max_scoring_round=1
        )
        assert result.score.originality.score == 0.7
        assert cost == 0.01

    def test_scoring_lack_information_retry(
        self,
        scoring_idea: ScoringIdea,
        idea: GeneratedIdea,
        mock_clients: tuple[Any, Any],
        mocker: Any,
    ) -> None:
        llm, db = mock_clients
        db.query.return_value = {"matches": []}

        # First response: lack information
        resp1 = mocker.MagicMock()
        resp1.content = '{"score": {}, "is_lack_information": true, "additional_key_sentences": ["K2"]}'
        resp1.cost = 0.01

        # Second response: success
        resp2 = mocker.MagicMock()
        resp2.content = '{"score": {"originality": {"score": 0.8, "reason": "r"}, "feasibility": {"score": 0.7, "reason": "r"}, "versatility": {"score": 0.6, "reason": "r"}}, "is_lack_information": false, "additional_key_sentences": []}'
        resp2.cost = 0.02

        llm.parse_chat.side_effect = [resp1, resp2]

        result, cost = scoring_idea.scoring_with_related_works(
            idea, max_scoring_round=2
        )

        assert result.score.originality.score == 0.8
        assert cost == 0.03
        assert llm.parse_chat.call_count == 2

        # Check that second round used the new key sentence
        assert db.query.call_args_list[1][0][0] == "K2"

    def test_format_fetch_paper(self, scoring_idea: ScoringIdea) -> None:
        mock_result = {
            "matches": [
                {
                    "id": "1",
                    "metadata": {"abstract": "abs", "chunk_text": "text"},
                }
            ]
        }
        papers = scoring_idea._format_fetch_paper(mock_result)
        assert len(papers) == 1
        assert "Paper Id: 1" in papers[0]
        assert "Abstract: abs" in papers[0]
        assert "Releted Chunk: text" in papers[0]
