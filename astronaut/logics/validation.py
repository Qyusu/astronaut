import ast
import copy
import json
import os
import py_compile
import re
import traceback
import warnings
from textwrap import dedent
from typing import Literal, Optional

import qxmt
import yaml
from langsmith import traceable
from loguru import logger
from pinecone import QueryResponse
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from astronaut.constants import (
    DEFAULT_MAX_RETRY,
    DRYRUN_CONFIG_PATH,
    GENERATED_CODE_DIRC,
    GENERATED_MODULE_ROOT,
    GPT_MAX_TOKENS,
)
from astronaut.db import PineconeClient
from astronaut.experiment_utils import save_generated_feature_map_code
from astronaut.llm import ChatClient
from astronaut.logics.generation import GenerateCode
from astronaut.prompts.generator import RetryGenerateFeatureMapCodePrompt
from astronaut.prompts.pennylane_validator import PennyLaneDocsValidatePrompt
from astronaut.schema import MESSAGE_HISTORY_TYPE, GeneratedResult, RunContext

TMP_CODE_PATH = "tmp_code.py"
DRYRUN_CODE_PATH = "dry_run.py"


class DocsValidateResult(BaseModel):
    class_name: str = Field(..., description="The class name")
    user_args_name: list[str] = Field(..., description="The arguments defined in the generated code.")
    docs_args_name: list[str] = Field(..., description="The arguments supported in the documentation.")


class DocsValidateResultList(BaseModel):
    result: list[DocsValidateResult] = Field(..., description="The results of the document validation.")


class CodeValidator:
    """A class for validating generated code.

    This class provides functionality to validate generated code by checking its syntax,
    Pennylane documentation, and ensuring it is not the same as the previous code.

    Args:
        code (str): The generated code to validate
        llm_client (ChatClient): Client for interacting with the language model

    Methods:
        validate_by_py_compile: Validates the code by checking its syntax
        validate_by_ast: Validates the code by checking its syntax
        validate_by_pennylane_doc: Validates the code by checking its Pennylane documentation
        validate_all: Validates the code by checking its syntax, Pennylane documentation,
            and ensuring it is not the same as the previous code
    """

    def __init__(self, code: str, llm_client: ChatClient, model_version: str, db_client: PineconeClient) -> None:
        self.source_code = code
        self.llm_client = llm_client
        self.model_version = model_version
        self.pennylane_db = db_client
        self.error_messages: list[str] = []
        self.total_cost = 0.0
        self.class_code = self._extract_class_code()
        self.qml_functions = self._extract_pennylane_function()
        self.qml_call_names = self._extract_pennylane_call_names()

    def _extract_class_code(self) -> str:
        class_pattern = r"(class \w+.*?:\n(?: {4}.*\n?)*)"
        matches = re.findall(class_pattern, self.source_code, re.DOTALL)

        if not matches:
            error_message = "Feature map class is not found."
            logger.info(error_message)
            self.error_messages.append(error_message)

        return matches[0]

    def _extract_pennylane_function(self) -> list[str]:
        functions = []
        start_idx = self.class_code.find("qml.")  # Start by looking for "qml." pattern
        while start_idx != -1:
            # Track the position and initialize a stack to manage parentheses
            end_idx = start_idx
            stack = []

            # Process each character after the "qml." pattern to balance parentheses
            while end_idx < len(self.class_code):
                char = self.class_code[end_idx]
                if char == "(":
                    stack.append(char)  # Push opening parenthesis
                elif char == ")":
                    if stack:
                        stack.pop()  # Pop closing parenthesis
                    else:
                        break
                    if not stack:  # If stack is empty, parentheses are balanced
                        functions.append(self.class_code[start_idx : end_idx + 1])
                        break
                end_idx += 1

            # Continue searching for the next "qml." pattern
            start_idx = self.class_code.find("qml.", end_idx)

        return list(set(functions))

    def _extract_pennylane_call_names(self) -> list[str]:
        matches = re.findall(r"qml\.\w+", self.source_code)
        return list(set(matches))

    @traceable(tags=["validation", "code"])
    def validate_by_py_compile(self) -> None:
        try:
            # write to tmp python file for compile
            with open(TMP_CODE_PATH, "w") as file:
                file.write(self.source_code)
            py_compile.compile(TMP_CODE_PATH, doraise=True)
            # remove tmp python file
            os.remove(TMP_CODE_PATH)
            logger.info("py_compile: Syntax is correct.")
        except py_compile.PyCompileError as e:
            error_message = f"py_compile: Syntax error: {e}"
            logger.info(error_message)
            self.error_messages.append(error_message)

    @traceable(tags=["validation", "code"])
    def validate_by_ast(self) -> None:
        try:
            ast.parse(self.source_code)
            logger.info("ast: Code is syntactically valid.")
        except SyntaxError as e:
            error_message = f"ast: Syntax error: {e}"
            logger.info(error_message)
            self.error_messages.append(error_message)

    def _construct_retrieved_docs_string(self, result: QueryResponse, docs_type: str) -> str:
        retrieved_docs_string = ""
        for i, r in enumerate(result["matches"]):
            if docs_type == "source_code":
                doc_template = dedent(
                    """
                    ## Reference {reference_num}
                    File Name: {file_name}

                    Code:
                    ```python
                    {code}
                    ```
                    ----------------------------------------------------------- \n
                    """
                )
                doc = doc_template.format(
                    reference_num=i + 1, file_name=r["metadata"]["file_path"], code=r["metadata"]["chunk_text"]
                )
            elif docs_type == "class_doc":
                doc_template = dedent(
                    """
                    ## Reference {reference_num}
                    File Name: {file_name}

                    Class Name: {class_name}

                    Docstring:
                    {doc_string}
                    ----------------------------------------------------------- \n
                    """
                )
                doc = doc_template.format(
                    reference_num=i + 1,
                    file_name=r["metadata"]["file_path"],
                    class_name=r["metadata"]["class_name"],
                    doc_string=r["metadata"]["chunk_text"],
                )
            else:
                raise ValueError(f"Invalid docs type: {docs_type}")

            retrieved_docs_string += doc
        return retrieved_docs_string

    def _format_docs_validation_result(self, content: str) -> DocsValidateResultList:
        adapter = TypeAdapter(DocsValidateResultList)
        try:
            validation_list = adapter.validate_python(json.loads(content))
        except ValidationError as e:
            logger.info("Validation error:", e)

        return validation_list

    def _format_error_messages(self, validation_list: DocsValidateResultList) -> list[str]:
        error_messages = []
        for result in validation_list.result:
            for user_arg in result.user_args_name:
                if user_arg not in result.docs_args_name:
                    error_message = (
                        f"{result.class_name}: Argument '{user_arg}' is not supported. "
                        f"Please only use supported arguments: {result.docs_args_name}"
                    )
                    error_messages.append(error_message)

        return error_messages

    @traceable(tags=["validation", "code"])
    def validate_by_pennylane_doc(
        self,
        docs_type: Literal["source_code", "class_doc"],
        docs_top_k: int,
        score_threshold: Optional[float] = None,
        strict: bool = True,
    ) -> None:
        functions_str = "- " + "\n- ".join(self.qml_functions)
        if strict:
            filter_list = []
            for call_name in self.qml_call_names:
                filter_list.append({"call_name": call_name})
            metadata_filter = {"$or": filter_list}
            docs_top_k = int(len(self.qml_call_names) * 1.5)
        else:
            metadata_filter = {}

        result = self.pennylane_db.query(functions_str, top_k=docs_top_k, metadata_filter=metadata_filter)
        if score_threshold is not None:
            result["matches"] = [r for r in result["matches"] if r["score"] >= score_threshold]

        retrieved_docs_string = self._construct_retrieved_docs_string(result, docs_type)
        system_prompt, user_promt = PennyLaneDocsValidatePrompt(
            methods=functions_str, references=retrieved_docs_string
        ).build()
        content, _, cost = self.llm_client.parse_chat(
            system_prompt=system_prompt,
            user_prompt=user_promt,
            message_history=[],
            temperature=0.0,
            n=1,
            max_tokens=GPT_MAX_TOKENS,
            response_format=DocsValidateResultList,
            model_version=self.model_version,
        )
        self.total_cost += cost

        validation_list = self._format_docs_validation_result(content)
        error_messages = self._format_error_messages(validation_list)
        if len(error_messages) == 0:
            logger.info("pennylane_doc: Feature map is valid.")
        else:
            logger.info(f"pennylane_doc: Feature map is invalid. Error Messages: {error_messages}")
            self.error_messages.extend(error_messages)

    def validate_all(
        self, docs_type: Literal["source_code", "class_doc"], docs_top_k: int, strict: bool = True
    ) -> tuple[bool, list[str], float]:
        self.validate_by_py_compile()
        self.validate_by_ast()
        self.validate_by_pennylane_doc(docs_type=docs_type, docs_top_k=docs_top_k, strict=strict)
        is_valid = True if len(self.error_messages) == 0 else False
        return is_valid, self.error_messages, self.total_cost


@traceable(tags=["validation", "code"])
def is_same_code(last_code: str, code: str) -> bool:
    """check if last code is same as current code

    Args:
        last_code (str): generated code from last iteration
        code (str): generated code from current iteration

    Returns:
        bool: True if last code is same as current code
    """
    return last_code == code


@traceable(tags=["validation", "code"])
def dry_run(experiment: qxmt.Experiment, config_path: str, generate_result: GeneratedResult) -> tuple[str, list[str]]:
    """Dry run the generated code by QXMT experiment.

    Args:
        experiment (qxmt.Experiment): qxmt experiment instance
        config_path (str): path to config file for dry run
        generate_result (GeneratedResult): generated result by LLM

    Returns:
        str: error message if code execution is failed
    """
    config = yaml.safe_load(open(config_path))
    config["feature_map"]["module_name"] = f"{GENERATED_MODULE_ROOT}.{experiment.name}.dry_run"
    config["feature_map"]["implement_name"] = generate_result.implement.class_name

    # save config in tmp file
    tmp_config_path = f"{config_path}".replace(".yaml", "_tmp.yaml")
    with open(tmp_config_path, "w") as file:
        yaml.dump(config, file)

    error_massage = ""
    warning_messages = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("default")
        try:
            logger.info("Execute dry run ...")
            experiment.run(config_source=tmp_config_path, add_results=False)
            os.remove(tmp_config_path)
            logger.info("dry run: Code execution is successful.")
        except Exception as e:
            logger.info(f"Error occurred during dry run: {e}")
            error_massage = traceback.format_exc()

        for warning in w:
            warning_messages.append(str(warning.message))

    return error_massage, warning_messages


@traceable(tags=["validation", "code"])
def validate_generated_code(
    llm_client: ChatClient,
    db_client: PineconeClient,
    context: RunContext,
    generate_result: GeneratedResult,
    experiment: qxmt.Experiment,
    max_retry: int = DEFAULT_MAX_RETRY,
    dry_run_config_path: str = DRYRUN_CONFIG_PATH,
) -> tuple[GeneratedResult, float]:
    """Validates generated code through multiple checks and retries if needed.

    The validation process includes:
    1. Syntax validation using py_compile and ast
    2. PennyLane documentation validation
    3. Code uniqueness check against previous code
    4. Dry run execution

    If validation fails, the code is regenerated up to max_retry times.
    The original idea and score are preserved throughout the process.

    Args:
        llm_client (ChatClient): Client for interacting with the language model
        db_client (PineconeClient): Client for Pinecone database
        context (RunContext): Context of the current run
        generate_result (GeneratedResult): Generated code result to validate
        experiment (qxmt.Experiment): QXMT experiment object for dry run
        max_retry (int, optional): Maximum number of retry attempts. Defaults to DEFAULT_MAX_RETRY.
        dry_run_config_path (str, optional): Path to dry run config. Defaults to DRYRUN_CONFIG_PATH.

    Raises:
        ValueError: If valid code cannot be generated after max_retry attempts

    Returns:
        tuple[GeneratedResult, float]: Validated code result and total validation cost
    """
    re_generate_code_result = copy.deepcopy(generate_result.implement)
    code_generator = GenerateCode(llm_client, context.model_versions.default, context.model_versions.parser)

    validation_cost = 0.0
    retry_num = 0
    validation_message_history: MESSAGE_HISTORY_TYPE = []
    while True:
        validator = CodeValidator(
            code=re_generate_code_result.code,
            llm_client=llm_client,
            model_version=context.model_versions.validation,
            db_client=db_client,
        )
        is_valid, error_messages, cost = validator.validate_all(docs_type="class_doc", docs_top_k=10, strict=True)
        validation_cost += cost
        if is_valid:
            validate_result = GeneratedResult(
                idea=generate_result.idea,
                score=generate_result.score,
                implement=re_generate_code_result,
            )
            break
        else:
            logger.info("Generated code is invalid. Retry to generate code...")
            system_prompt, user_prompt = RetryGenerateFeatureMapCodePrompt(
                code=re_generate_code_result.code, error_messages=error_messages, warning_messages=[]
            ).build()
            re_generate_code_result, validation_message_history, cost = code_generator.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=validation_message_history,
                n_history=None,
            )
            validation_cost += cost
            retry_num += 1

        if retry_num > max_retry:
            raise ValueError("Failed to generate valid code.")

    # if generated code is same as last code, stop validation and experiment
    if is_same_code(last_code=context.last_code, code=validate_result.implement.code):
        logger.info("Generated code is same as last code.")
        return (
            GeneratedResult(
                idea=validate_result.idea,
                score=validate_result.score,
                implement=validate_result.implement,
            ),
            validation_cost,
        )

    # dry run
    for _ in range(max_retry):
        save_generated_feature_map_code(
            f"{GENERATED_CODE_DIRC}/{experiment.name}/{DRYRUN_CODE_PATH}", validate_result.implement.code
        )
        error_massage, warning_messages = dry_run(
            experiment=experiment, config_path=dry_run_config_path, generate_result=validate_result
        )
        if error_massage == "":
            break
        else:
            system_prompt, user_prompt = RetryGenerateFeatureMapCodePrompt(
                code=validate_result.implement.code, error_messages=[error_massage], warning_messages=warning_messages
            ).build()
            re_generate_code_result, validation_message_history, cost = code_generator.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                message_history=validation_message_history,
                n_history=None,
            )
            validate_result = GeneratedResult(
                idea=generate_result.idea,
                score=generate_result.score,
                implement=re_generate_code_result,
            )

    return validate_result, validation_cost
