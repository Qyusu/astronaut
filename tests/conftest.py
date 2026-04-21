import shutil
from pathlib import Path
from typing import Generator

import pytest

from astronaut.llm import ChatClient
from astronaut.logics.common.review import ReviewIdea
from astronaut.schema import ModelVersions


@pytest.fixture(scope="module")
def model_versions() -> ModelVersions:
    return ModelVersions(
        default="gpt-5",
        expanding="gpt-5",
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


@pytest.fixture(scope="session", autouse=True)
def cleanup_magicmock_dirs() -> Generator[None, None, None]:
    """Clean up directories starting with <MagicMock or <Mock generated during tests."""
    yield
    generated_dir = Path(__file__).parent.parent / "astronaut" / "generated"
    if generated_dir.exists() and generated_dir.is_dir():
        for path in generated_dir.iterdir():
            if (path.name.startswith("<MagicMock")) or (
                path.name.startswith("<Mock")
            ):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
