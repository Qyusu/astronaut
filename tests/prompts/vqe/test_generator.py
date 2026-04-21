from astronaut.prompts.vqe.generator import (
    BaseGenerateAnsatzIdeaPrompt,
    ExpandingAnsatzIdeaPrompt,
    FeedbackBasedAnsatzIdeaPrompt,
    GenerateAnsatzCodePrompt,
    ResearchBasedAnsatzIdeaPrompt,
)


class TestVQEGeneratorPrompts:
    def test_base_ansatz_idea_prompt(self) -> None:
        prompt = BaseGenerateAnsatzIdeaPrompt(
            ansatz_constraints="Constraints",
            molecule_hamiltonian_definition="Hamiltonian",
            cost_function_definition="Cost Function",
            optimizer_settings="Optimizer",
            target_molecule_names="Molecule",
        )
        developer_prompt = prompt._build_developer_prompt()
        assert developer_prompt["role"] == "developer"
        assert (
            "# VQE Ansatz Constraints\nConstraints"
            in developer_prompt["content"]
        )
        assert "# Hamiltonian\nHamiltonian" in developer_prompt["content"]
        assert (
            "# Cost Function Definition\nCost Function"
            in developer_prompt["content"]
        )
        assert (
            "The cost function is calculated based on the Molecule molecules."
            in developer_prompt["content"]
        )
        assert "# Optimizer Settings\nOptimizer" in developer_prompt["content"]

    def test_research_based_ansatz_idea_prompt(self) -> None:
        prompt = ResearchBasedAnsatzIdeaPrompt()
        developer_prompt, user_prompt = prompt.build(
            idea_num=3, research_report="Valid report"
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "# Research Report\nValid report" in user_prompt["content"]

    def test_expanding_ansatz_idea_prompt(self) -> None:
        prompt = ExpandingAnsatzIdeaPrompt()
        developer_prompt, user_prompt = prompt.build(
            idea_num=3, experiment_report="Exp report"
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "# Experiment Report\nExp report" in user_prompt["content"]

    def test_feedback_based_ansatz_idea_prompt(self) -> None:
        prompt = FeedbackBasedAnsatzIdeaPrompt()
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

    def test_generate_ansatz_code_prompt(self) -> None:
        prompt = GenerateAnsatzCodePrompt(
            code="Old code", idea="New Idea", pennylane_reference="Refs"
        )
        developer_prompt, user_prompt = prompt.build()
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "Old code" in developer_prompt["content"]
        assert "# Idea\nNew Idea" in user_prompt["content"]
        assert "Refs" in user_prompt["content"]
