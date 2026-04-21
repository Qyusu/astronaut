from astronaut.prompts.common.pennylane_reference import RelatedReferencePrompt


class TestRelatedReferencePrompt:
    def test_related_reference_prompt(self) -> None:
        prompt = RelatedReferencePrompt(reference="Ref list", idea="My Idea")
        developer_prompt, user_prompt = prompt.build()
        assert developer_prompt["role"] == "developer"
        assert user_prompt["role"] == "user"
        assert "---\nRef list\n---" in developer_prompt["content"]
        assert "My Idea" in user_prompt["content"]
