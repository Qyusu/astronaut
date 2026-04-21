from astronaut.prompts.common.scoring import ScoringIdeaPrompt


class TestScoringIdeaPrompt:
    def test_scoring_idea_prompt(self) -> None:
        prompt = ScoringIdeaPrompt(score_histories="")
        # Round 1
        developer_prompt, user_prompt = prompt.build(
            idea="My Idea", related_work="Related work 1", round=1, max_round=3
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "Round 1/3." in user_prompt["content"]
        assert "# Proposed Idea\nMy Idea" in user_prompt["content"]
        assert "# Related Work\nRelated work 1" in user_prompt["content"]

        # Intermediate Round
        developer_prompt, user_prompt = prompt.build(
            idea="My Idea", related_work="Related work 2", round=2, max_round=3
        )
        assert "Round 2/3." in user_prompt["content"]
        assert "# Proposed Idea\nMy Idea" not in user_prompt["content"]
        assert "# Related Work\nRelated work 2" in user_prompt["content"]

        # Final Round
        developer_prompt, user_prompt = prompt.build(
            idea="My Idea", related_work="Related work 3", round=3, max_round=3
        )
        assert "Round 3/3 (Final Round)." in user_prompt["content"]
        assert "# Related Work\nRelated work 3" in user_prompt["content"]

    def test_scoring_idea_prompt_with_score_history(self) -> None:
        prompt = ScoringIdeaPrompt(score_histories="History")
        developer_prompt, user_prompt = prompt.build(
            idea="My Idea", related_work="Related work 1", round=1, max_round=3
        )
        assert "History" in developer_prompt["content"]
