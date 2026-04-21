import json
import os
from typing import Literal

from langsmith import traceable
from loguru import logger
from openai.types.shared.reasoning_effort import ReasoningEffort

import astronaut.constants as exp_const
from astronaut.llm import ChatClient
from astronaut.prompts.common import RelatedReferencePrompt
from astronaut.schema import PennyLaneReferenceResult

NOT_DEFINED = "Not defined"
IGNORE_METHODS = ["qml.qnode", "qml.QNode", "qml.device", "qml.Device"]


class PennyLaneReference:
    def __init__(
        self,
        client: ChatClient | None,
        model_version: str,
        reference_path: str,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Literal["low", "medium", "high"] | None = None,
    ) -> None:
        self.client = client
        self.model_version = model_version
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.reference = self._load_reference(reference_path)
        self.simple_reference_str = self._get_simple_reference_str()

    def _load_reference(
        self, path: str
    ) -> dict[str, dict[str, list[dict[str, str]]]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference file not found: {path}")

        with open(path, "r") as f:
            return json.load(f)

    def _get_simple_reference_str(self) -> str:
        reference_str = ""
        for k, v in self.reference.items():
            reference_str += f'- "{k}": {v["description"]}\n'
        return reference_str

    def get_full_reference(self) -> dict[str, dict[str, list[dict[str, str]]]]:
        return self.reference

    @traceable(tags=exp_const.LANGSMITH_PENNYLANE_REFERENCE_TAGS)
    def extract_related_method_names(self, idea: str) -> list[str]:
        if self.client is None:
            raise ValueError(
                "PennyLaneReference class is not initialized with a client."
            )

        developer_prompt, user_prompt = RelatedReferencePrompt(
            reference=self.simple_reference_str, idea=idea
        ).build()

        logger.info("Extracting PennyLane reference...")
        response = self.client.parse_chat(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=[],
            n_history=0,
            response_format=PennyLaneReferenceResult,
            reasoning=self.reasoning_effort,
            verbosity=self.verbosity,
            model_version=self.model_version,
        )

        result = json.loads(response.content)
        logger.debug(f"PennyLane reference result: {result}")
        logger.info("PennyLane reference extraction is done.")

        return PennyLaneReferenceResult(**result).method_names

    def extract_related_reference_str(self, idea: str) -> str:
        method_names = self.extract_related_method_names(idea)
        filtered_method_names = [
            method for method in method_names if method not in IGNORE_METHODS
        ]

        references = []
        for method in filtered_method_names:
            method_doc = self.reference.get(method, {})
            if not method_doc:
                continue

            method_desc = method_doc.get("description", NOT_DEFINED)

            args_str = ""
            for arg in method_doc.get("args", []):
                name = arg.get("name", None)
                type = arg.get("type", NOT_DEFINED)
                required = arg.get("required", NOT_DEFINED)
                description = arg.get("description", NOT_DEFINED)
                if name is not None:
                    arg_str = f'- "{name}" ({type}, required={required}): {description}'
                    args_str += arg_str + "\n"

            method_str = (
                f'"{method}":\nDescription: {method_desc}\nArgs:\n{args_str}'
            )
            references.append(method_str)

        return "\n-------------\n".join(references)
