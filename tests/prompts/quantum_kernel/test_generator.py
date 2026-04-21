from astronaut.prompts.quantum_kernel.generator import (
    BaseGenerateFeatureMapIdeaPrompt,
    ExpandingFeatureMapIdeaPrompt,
    FeedbackBasedFeatureMapIdeaPrompt,
    GenerateFeatureMapCodePrompt,
    ResearchBasedFeatureMapIdeaPrompt,
)


class TestQuantumKernelGeneratorPrompts:
    def test_base_feature_map_idea_prompt(self) -> None:
        prompt = BaseGenerateFeatureMapIdeaPrompt(
            feature_map_constraints="Constraints",
            performance_metric_definition="Metric",
            input_data_constraints="Data",
        )
        developer_prompt = prompt._build_developer_prompt()
        assert developer_prompt["role"] == "developer"
        assert (
            "# Feature Map Constraints\nConstraints"
            in developer_prompt["content"]
        )
        assert (
            "# Performance Metric Definition\nMetric"
            in developer_prompt["content"]
        )
        assert "# Input Data Constraints\nData" in developer_prompt["content"]

    def test_research_based_feature_map_idea_prompt(self) -> None:
        prompt = ResearchBasedFeatureMapIdeaPrompt()
        developer_prompt, user_prompt = prompt.build(
            idea_num=3, research_report="Valid report"
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "# Research Report\nValid report" in user_prompt["content"]

    def test_expanding_feature_map_idea_prompt(self) -> None:
        prompt = ExpandingFeatureMapIdeaPrompt()
        developer_prompt, user_prompt = prompt.build(
            idea_num=3, experiment_report="Exp report"
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "# Experiment Report\nExp report" in user_prompt["content"]

    def test_feedback_based_feature_map_idea_prompt(self) -> None:
        prompt = FeedbackBasedFeatureMapIdeaPrompt()
        # With seed ideas
        developer_prompt, user_prompt = prompt.build(
            current_trial_num=1,
            max_trial_num=5,
            idea_num=3,
            review_comment="Comment 1",
            seed_ideas="Seed 1",
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "Trial 1/5." in user_prompt["content"]
        assert "**3 improved ideas**" in user_prompt["content"]
        assert (
            "## Feedback from the Previous Trial\nComment 1"
            in user_prompt["content"]
        )
        assert "multiple seed ideas:\nSeed 1" in user_prompt["content"]

        # No seed ideas
        dev, user = prompt.build(
            current_trial_num=2,
            max_trial_num=5,
            idea_num=3,
            review_comment="Comment 2",
            seed_ideas=None,
        )
        assert "# Seed Ideas" not in user["content"]

        # No review comment
        dev, user = prompt.build(
            current_trial_num=2,
            max_trial_num=5,
            idea_num=3,
            review_comment="",
            seed_ideas=None,
        )
        assert "## Feedback from the Previous Trial" not in user["content"]

    def test_generate_feature_map_code_prompt(self) -> None:
        prompt = GenerateFeatureMapCodePrompt(
            code="Old code",
            idea="New Idea",
            pennylane_reference="Refs",
            input_data_constraints="Constraints",
        )
        developer_prompt, user_prompt = prompt.build()
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "Constraints" in developer_prompt["content"]
        assert "Old code" in developer_prompt["content"]
        assert "# Idea\nNew Idea" in user_prompt["content"]
        assert "Refs" in user_prompt["content"]
