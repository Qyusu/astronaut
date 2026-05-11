import pytest

from astronaut.llm import ChatClient
from astronaut.logics.common.review import ReviewIdea
from astronaut.schema import ModelVersions


@pytest.fixture(scope="module")
def model_versions() -> ModelVersions:
    return ModelVersions(
        default="gpt-5",
        filtering="gpt-5",
        idea="gpt-5",
        scoring="gpt-5",
        summary="gpt-5",
        expert="gpt-5",
        reflection="gpt-5",
        reference="gpt-5",
        code="gpt-5",
        validation="gpt-5",
        review="gpt-5",
    )


@pytest.fixture(scope="module")
def llm_client(model_versions: ModelVersions) -> ChatClient:
    return ChatClient(
        platform="openai",
        api_key="API_KEY",
        default_model_version=model_versions.default,
    )


@pytest.fixture(scope="module")
def review_idea(
    llm_client: ChatClient, model_versions: ModelVersions
) -> ReviewIdea:
    return ReviewIdea(
        client=llm_client,
        model_version=model_versions.review,
    )
