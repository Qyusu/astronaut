from astronaut.prompts.vqe.review import ReviewIdeaPrompt


class TestVQEReviewPrompt:
    def test_review_idea_prompt_with_target_molecule_names(self) -> None:
        prompt = ReviewIdeaPrompt(
            ansatz_constraints="Ansatz Constraints",
            cost_function_definition="Cost Function Definition",
            optimizer_settings="Optimizer Settings",
            target_molecule_names="H2, LiH",
            max_suggestion_num=3,
        )
        developer_prompt, user_prompt = prompt.build(
            last_trial_num=1, last_trial_results="Results"
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert (
            "# VQE Ansatz Constraints\nAnsatz Constraints"
            in developer_prompt["content"]
        )
        assert (
            "# Cost Function Definition\nCost Function Definition"
            in developer_prompt["content"]
        )
        assert (
            "The cost function is calculated based on the H2, LiH molecules."
            in developer_prompt["content"]
        )
        assert (
            "# Optimizer Settings\nOptimizer Settings"
            in developer_prompt["content"]
        )
        assert (
            "- Limit the number of suggestions to 3 or fewer."
            in developer_prompt["content"]
        )
        assert (
            "# Previous Trial Idea and Results (Trial Number: 1)\nResults"
            in user_prompt["content"]
        )

    def test_review_idea_prompt_no_target_molecule_names(self) -> None:
        prompt = ReviewIdeaPrompt(
            ansatz_constraints="Ansatz Constraints",
            cost_function_definition="Cost Function Definition",
            optimizer_settings="Optimizer Settings",
            target_molecule_names="",
            max_suggestion_num=3,
        )
        developer_prompt, user_prompt = prompt.build(
            last_trial_num=1, last_trial_results="Results"
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert (
            "# Cost Function Definition\nCost Function Definition"
            in developer_prompt["content"]
        )
        assert (
            "The cost function is calculated based on"
            not in developer_prompt["content"]
        )
