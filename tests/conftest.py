import pytest

from astronaut.llm import ChatClient
from astronaut.logics.common.review import ReviewIdea
from astronaut.schema import ModelVersions


@pytest.fixture(scope="module")
def model_versions() -> ModelVersions:
    return ModelVersions(
        default="gpt-4o",
        idea="gpt-4o",
        scoring="gpt-4o",
        summary="gpt-4o",
        reflection="gpt-4o",
        code="gpt-4o",
        validation="gpt-4o",
        review="gpt-4o",
        parser="gpt-4o-mini",
    )


@pytest.fixture(scope="module")
def llm_client(model_versions: ModelVersions) -> ChatClient:
    return ChatClient(
        platform="openai",
        api_key="API_KEY",
        default_model_version=model_versions.default,
    )


@pytest.fixture(scope="module")
def review_idea(llm_client: ChatClient, model_versions: ModelVersions) -> ReviewIdea:
    return ReviewIdea(
        client=llm_client,
        model_version=model_versions.review,
        parser_model_version=model_versions.parser,
    )
