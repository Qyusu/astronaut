from astronaut.prompts.common.reflection import (
    DebateAdvocateIdeaPrompt,
    DebateCriticIdeaPrompt,
    SelfReflectionIdeaPrompt,
)


class TestReflectionPrompts:
    def test_self_reflection_idea_prompt(self) -> None:
        prompt = SelfReflectionIdeaPrompt(
            expertise="Physics", constraints="Only use linear transformations"
        )
        developer_prompt, user_prompt = prompt.build(
            current_reflection_round=1,
            max_reflection_round=2,
            previous_idea="Old Idea",
            previous_score="Originality: 0.5",
            related_work="Some related work",
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "**Physics**" in developer_prompt["content"]
        assert (
            "# Constraints\nOnly use linear transformations"
            in developer_prompt["content"]
        )
        assert "Round 1/2." in user_prompt["content"]
        assert "# Previous Idea\nOld Idea" in user_prompt["content"]
        assert (
            "# First Round Score (Only updates in the first round):\nOriginality: 0.5"
            in user_prompt["content"]
        )
        assert "# Related Work\nSome related work" in user_prompt["content"]

    def test_debate_advocate_idea_prompt(self) -> None:
        prompt = DebateAdvocateIdeaPrompt(
            expertise="Math", constraints="Only use linear transformations"
        )
        developer_prompt, user_prompt = prompt.build(
            current_reflection_round=1,
            max_reflection_round=2,
            previous_idea="Initial idea",
            critical_feedback="Consider circuit depth",
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "**Math**" in developer_prompt["content"]
        assert (
            "# Constraints\nOnly use linear transformations"
            in developer_prompt["content"]
        )
        assert "Round 1/2." in user_prompt["content"]
        assert "# Original Idea\nInitial idea" in user_prompt["content"]
        assert (
            "# Critical Feedback\nConsider circuit depth"
            in user_prompt["content"]
        )

    def test_debate_critic_idea_prompt_with_query(self) -> None:
        prompt = DebateCriticIdeaPrompt(
            expertise="Chemist",
            constraints="Only use linear transformations",
            max_feedback_num=3,
        )
        developer_prompt, user_prompt = prompt.build(
            current_reflection_round=1,
            max_reflection_round=2,
            previous_idea="Initial idea",
            previous_score="0.8",
            advocate_response="It is good",
            additional_information="Some info",
            additional_information_query="Query",
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "**Chemist**" in developer_prompt["content"]
        assert (
            "# Constraints\nOnly use linear transformations"
            in developer_prompt["content"]
        )
        assert (
            "[feedback_1, feedback_2, ..., feedback_3]"
            in developer_prompt["content"]
        )
        assert (
            "The maximum number of feedback is 3."
            in developer_prompt["content"]
        )
        assert "# Your Question for the Expert\nQuery" in user_prompt["content"]
        assert "# Expert Response\nSome info" in user_prompt["content"]

    def test_debate_critic_idea_prompt_no_query(self) -> None:
        prompt = DebateCriticIdeaPrompt(expertise="Chemist", constraints="None")
        developer_prompt, user_prompt = prompt.build(
            current_reflection_round=1,
            max_reflection_round=2,
            previous_idea="Initial idea",
            previous_score="0.8",
            advocate_response="It is good",
            additional_information="Some info",
            additional_information_query=None,
        )
        assert "Round 1/2." in user_prompt["content"]
        assert "# Improved Idea\nInitial idea" in user_prompt["content"]
        assert (
            "# First Round Score (Only updates in the first round):\n0.8"
            in user_prompt["content"]
        )
        assert (
            "# Response to Previous Criticism\nIt is good"
            in user_prompt["content"]
        )
        assert "# Related Work\nSome info" in user_prompt["content"]
