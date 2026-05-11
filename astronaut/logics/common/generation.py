import json
import time
import traceback
from typing import Any, Callable, Iterable, Literal

from langsmith import traceable
from loguru import logger
from openai.types.responses.tool_param import ParseableToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort

from astronaut.constants import PENNYLANE_REFERENCE_PATH
from astronaut.llm import ChatClient
from astronaut.logics.common.pennylane_reference import PennyLaneReference
from astronaut.params import PennyLaneReferenceParams
from astronaut.schema import (
    MESSAGE_HISTORY_TYPE,
    GeneratedIdeaResult,
    GeneratedImpl,
    GeneratedImplResult,
)


class GenerateIdea:
    """A class for generating quantum feature map ideas using LLM.

    This class provides functionality to generate ideas for quantum feature maps
    using language models. It supports both standard and reasoning-based model
    versions, with appropriate parsing and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use

    Methods:
        generate: Generates feature map ideas using the language model
    """

    def __init__(
        self,
        client: ChatClient,
        model_version: str,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        self.client = client
        self.model_version = model_version
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    @traceable(tags=["generation", "idea"])
    def generate(
        self,
        developer_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: int | None,
    ) -> tuple[GeneratedIdeaResult, MESSAGE_HISTORY_TYPE, float]:
        logger.info("Generating idea ...")
        logger.debug(
            f"developer_prompt: {developer_prompt}, user_prompt: {user_prompt}"
        )
        result, message_history, cost = self._generate_ideas(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=message_history,
            n_history=n_history,
        )

        logger.debug(f"Generated idea result: {result}")
        logger.info("Generated idea is done and result is parsed as JSON.")

        return result, message_history, cost

    def _generate_ideas(
        self,
        developer_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: int | None,
        model_version: str | None = None,
    ) -> tuple[GeneratedIdeaResult, MESSAGE_HISTORY_TYPE, float]:
        response = self.client.parse_chat(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=message_history,
            n_history=n_history,
            response_format=GeneratedIdeaResult,
            reasoning=self.reasoning_effort,
            verbosity=self.verbosity,
            model_version=model_version or self.model_version,
        )

        result = json.loads(response.content)

        return (
            GeneratedIdeaResult(**result),
            response.message_history,
            response.cost,
        )


class GenerateCode:
    """A class for generating quantum feature map code using LLM.

    This class provides functionality to generate Python code for quantum feature maps
    using language models. It supports both standard and reasoning-based model
    versions, with appropriate parsing and response formatting.

    Args:
        client (ChatClient): Client for interacting with the language model
        model_version (str): Version of the language model to use

    Methods:
        generate: Generates feature map code using the language model
    """

    def __init__(
        self,
        client: ChatClient,
        code_model_version: str,
        reference_model_version: str,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        self.client = client
        self.model_version = code_model_version
        self.reference_model_version = reference_model_version
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.pennylane_reference = PennyLaneReference(
            client=client,
            model_version=reference_model_version,
            reference_path=PENNYLANE_REFERENCE_PATH,
            reasoning_effort=PennyLaneReferenceParams().reasoning_effort,
            verbosity=PennyLaneReferenceParams().verbosity,
        )

    @traceable(tags=["generation", "code"])
    def generate_codes(
        self,
        prompt_builder_fn: Callable[
            [str, Any], tuple[dict[str, str], dict[str, str]]
        ],
        last_code: str,
        finalized_idea: GeneratedIdeaResult,
        message_history: MESSAGE_HISTORY_TYPE,
        n_code_history: int | None = None,
    ) -> tuple[GeneratedImplResult, MESSAGE_HISTORY_TYPE, float]:
        """
        Generate codes for a batch of ideas using a provided prompt builder function.

        Args:
            prompt_builder_fn: Function to build (developer_prompt, user_prompt) from last_code and idea_result.
            last_code: The previous code string.
            finalized_idea: GeneratedIdeaResult containing ideas to generate code for.
            message_history: Message history for the code generation.
            n_code_history: Number of code history messages to use.

        Returns:
            (GeneratedImplResult, updated_message_history, total_cost)
        """
        generated_results = []
        total_cost = 0.0
        for i, idea_result in enumerate(finalized_idea.results):
            logger.info(
                f"Generating code for idea ({i + 1}/{len(finalized_idea.results)})..."
            )
            pennylane_reference = (
                self.pennylane_reference.extract_related_reference_str(
                    idea=idea_result.get_string_for_code_generation()
                )
            )
            developer_prompt, user_prompt = prompt_builder_fn(
                last_code, idea_result, pennylane_reference
            )
            for _ in range(3):
                try:
                    generate_code_result, message_history, cost = self.generate(
                        developer_prompt=developer_prompt,
                        user_prompt=user_prompt,
                        message_history=message_history,
                        n_history=n_code_history,
                    )
                    total_cost += cost
                    break
                except Exception as e:
                    logger.debug(traceback.format_exc())
                    logger.warning(f"Raise Exception: {e}")
                    logger.info("Retry to generate code...")
                    time.sleep(3)

            # filtered_code_message_history = self._filter_message_history(
            #     code_message_history,
            #     message_idx=-2,  # latest user message
            #     keyword="# PennyLane Reference",
            # )
            # message_history = filtered_code_message_history
            generated_results.append(generate_code_result)

        return (
            GeneratedImplResult(results=generated_results),
            message_history,
            total_cost,
        )

    def generate(
        self,
        developer_prompt: dict[str, str],
        user_prompt: dict[str, str],
        message_history: MESSAGE_HISTORY_TYPE,
        n_history: int | None = None,
        model_version: str | None = None,
        tools: Iterable[ParseableToolParam] | None = None,
    ) -> tuple[GeneratedImpl, MESSAGE_HISTORY_TYPE, float]:
        logger.info("Generating code...")
        logger.debug(
            f"developer_prompt: {developer_prompt}, user_prompt: {user_prompt}"
        )

        response = self.client.parse_chat(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=message_history,
            n_history=n_history,
            response_format=GeneratedImpl,
            model_version=model_version or self.model_version,
            reasoning=self.reasoning_effort,
            verbosity=self.verbosity,
            tools=tools,
        )

        result = json.loads(response.content)
        logger.debug(f"Generated code result: {result}")
        logger.info("Generated code is done and result is parsed as JSON.")

        return GeneratedImpl(**result), response.message_history, response.cost

    def _filter_message_history(
        self,
        message_history: MESSAGE_HISTORY_TYPE,
        message_idx: int,
        keyword: str,
    ) -> MESSAGE_HISTORY_TYPE:
        """Remove PennyLane reference from message history of user prompt.
        This is because the reference is too long and it will cause the token limit error.

        Args:
            message_history (MESSAGE_HISTORY_TYPE): Message history to filter.
            message_idx (int): Index of the target message in message history.
            keyword (str): Keyword to filter.

        Returns:
            MESSAGE_HISTORY_TYPE: Filtered message history.
        """
        message = message_history[message_idx]
        filtering_idx = message["content"].find(keyword)  # type: ignore
        message["content"] = message["content"][:filtering_idx]  # type: ignore

        message_history[message_idx] = message  # type: ignore

        return message_history
