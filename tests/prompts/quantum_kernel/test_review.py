from astronaut.prompts.quantum_kernel.review import ReviewIdeaPrompt


class TestQuantumKernelReviewPrompt:
    def test_review_idea_prompt(self) -> None:
        prompt = ReviewIdeaPrompt(
            feature_map_constraints="Feature Map Constraints",
            performance_metric_definition="Performance Metric",
            input_data_constraints="Data Constraints",
            max_suggestion_num=3,
        )
        developer_prompt, user_prompt = prompt.build(
            last_trial_num=1, last_trial_results="Results"
        )
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert (
            "# Feature Map Constraints\nFeature Map Constraints"
            in developer_prompt["content"]
        )
        assert (
            "# Performance Metric Definition\nPerformance Metric"
            in developer_prompt["content"]
        )
        assert (
            "# Input Data Constraints\nData Constraints"
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
