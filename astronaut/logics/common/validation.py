import ast
import copy
import json
import os
import py_compile
import tempfile
import traceback
import warnings
from pathlib import Path
from traceback import TracebackException, format_exc

import qxmt
import yaml
from langsmith import traceable
from loguru import logger
from pydantic import BaseModel, Field

import astronaut.constants as exp_const
from astronaut.db import PineconeClient
from astronaut.llm import ChatClient
from astronaut.logics.common.code_utils import save_generated_code
from astronaut.logics.common.generation import GenerateCode
from astronaut.params import CodeValidationParams
from astronaut.prompts.common import FixCodePrompt
from astronaut.schema import MESSAGE_HISTORY_TYPE, GeneratedResult, RunContext

DRYRUN_CODE_PATH = "dry_run.py"


class DocsValidateResult(BaseModel):
    class_name: str = Field(..., description="The class name")
    user_args_name: list[str] = Field(
        ..., description="The arguments defined in the generated code."
    )
    docs_args_name: list[str] = Field(
        ..., description="The arguments supported in the documentation."
    )


class DocsValidateResultList(BaseModel):
    result: list[DocsValidateResult] = Field(
        ..., description="The results of the document validation."
    )


class ValidationResult(BaseModel):
    """Validation result container."""

    is_valid: bool = Field(..., description="Whether the validation passed")
    errors: list[str] = Field(
        default_factory=list, description="List of error messages"
    )
    cost: float = Field(default=0.0, description="Validation cost")

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.cost += other.cost


class SyntaxValidator:
    """Validates code syntax using py_compile and ast."""

    @traceable(tags=exp_const.LANGSMITH_VALIDATION_SYNTAX_TAGS)
    def validate_by_py_compile(self, source_code: str) -> ValidationResult:
        """Validate code syntax using py_compile with safe temporary file handling."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp_file:
                tmp_file.write(source_code)
                tmp_file_path = tmp_file.name

            try:
                py_compile.compile(tmp_file_path, doraise=True)
                logger.info("py_compile: Syntax is correct.")
                return ValidationResult(is_valid=True, errors=[])
            finally:
                # Ensure tmp file is always removed
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
        except Exception as e:
            logger.debug(traceback.format_exc())
            error_message = f"py_compile: {type(e).__name__}: {e}"
            logger.warning(error_message)
            return ValidationResult(is_valid=False, errors=[error_message])

    @traceable(tags=exp_const.LANGSMITH_VALIDATION_SYNTAX_TAGS)
    def validate_by_ast(self, source_code: str) -> ValidationResult:
        """Validate code syntax using ast."""
        try:
            ast.parse(source_code)
            logger.info("ast: Code is syntactically valid.")
            return ValidationResult(is_valid=True, errors=[])
        except Exception as e:
            logger.debug(traceback.format_exc())
            error_message = f"ast: {type(e).__name__}: {e}"
            logger.warning(error_message)
            return ValidationResult(is_valid=False, errors=[error_message])

    def validate(self, source_code: str) -> ValidationResult:
        """Validate code syntax using both py_compile and ast."""
        result = ValidationResult(is_valid=True, errors=[])

        py_compile_result = self.validate_by_py_compile(source_code)
        result.merge(py_compile_result)

        ast_result = self.validate_by_ast(source_code)
        result.merge(ast_result)

        return result


class PennyLaneDocValidator:
    """Validates code against PennyLane documentation."""

    def __init__(self, reference_path: str) -> None:
        self.reference_path = reference_path

    def validate(
        self,
        source_code: str,
        method_prefixes: list[str] = exp_const.PENNYLANE_METHOD_PREFIXES,
    ) -> ValidationResult:
        reference = self._get_pennylane_reference()
        methods = self._extract_pennylane_functions(
            source_code, method_prefixes
        )

        errors = []
        for method in methods:
            method_errors = []
            method_name = self._extract_method_name(method)
            signature = reference.get(method_name)
            if signature is None:
                method_errors.append("Not found in PennyLane Library")
            else:
                expected_args = signature["args"]
                args_erros = self._validate_args(
                    method, expected_args, method_prefixes
                )
                method_errors.extend(args_erros)

            if method_errors:
                method_errors_str = ", ".join(method_errors)
                errors.append(f"Method '{method_name}': {method_errors_str}")

        if len(errors) == 0:
            logger.info(
                f"pennylane_doc: {len(methods)} methods checked. All methods are valid."
            )
            return ValidationResult(is_valid=True, errors=[], cost=0.0)
        else:
            logger.warning(
                f"pennylane_doc: {len(methods)} methods checked. {len(errors)} errors found."
            )
            return ValidationResult(is_valid=False, errors=errors, cost=0.0)

    def _get_pennylane_reference(
        self,
    ) -> dict[str, dict[str, list[dict[str, str]]]]:
        if not os.path.exists(self.reference_path):
            raise FileNotFoundError(
                f"Reference file not found: {self.reference_path}"
            )
        with open(self.reference_path) as f:
            return json.load(f)

    def _extract_pennylane_functions(
        self, source_code: str, method_prefixes: list[str]
    ) -> list[str]:
        """Extract PennyLane function calls from class code using AST."""
        functions = []

        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            target = self._dotted_name(node.func)
            if target is None:
                continue

            if not self._is_supported_pennylane_method(target, method_prefixes):
                continue

            # Try to get the function call from the source code
            func_call = ast.get_source_segment(source_code, node)
            if func_call:
                functions.append(func_call)
                continue

            # Fallback: reconstruct the function call
            func_call = self._reconstruct_func_call(node, target)
            functions.append(func_call)

        return list(set(functions))

    def _dotted_name(self, expr: ast.AST) -> str | None:
        parts = []
        while isinstance(expr, ast.Attribute):
            parts.append(expr.attr)
            expr = expr.value
        if isinstance(expr, ast.Name):
            parts.append(expr.id)
            return ".".join(reversed(parts))

        return None

    def _is_supported_pennylane_method(
        self, target: str, method_prefixes: list[str]
    ) -> bool:
        parts = target.split(".")
        if len(parts) <= 1:
            return False

        return ".".join(parts[:-1]) in method_prefixes

    def _reconstruct_func_call(self, node: ast.AST, target: str) -> str:
        if not isinstance(node, ast.Call):
            raise ValueError("node must be an ast.Call")

        args_parts = []
        # Add positional arguments
        for arg in node.args:
            if isinstance(arg, ast.Constant):
                args_parts.append(repr(arg.value))
            elif isinstance(arg, ast.Name):
                args_parts.append(arg.id)
            else:
                args_parts.append("...")  # Complex expression placeholder

        # Add keyword arguments
        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Constant):
                args_parts.append(f"{keyword.arg}={repr(keyword.value.value)}")
            elif isinstance(keyword.value, ast.Name):
                args_parts.append(f"{keyword.arg}={keyword.value.id}")
            else:
                args_parts.append(f"{keyword.arg}=...")

        return f"{target}({', '.join(args_parts)})"

    def _extract_method_name(self, method_str: str) -> str:
        tree = ast.parse(method_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                return self._dotted_name(node.func) or ""

        return ""

    def _validate_args(
        self,
        method: str,
        expected_args: list[dict[str, str]],
        method_prefixes: list[str],
    ) -> list[str]:
        errors: list[str] = []
        expected_args_names = [arg["name"] for arg in expected_args]
        required_args = [arg for arg in expected_args if bool(arg["required"])]

        tree = ast.parse(method)
        expr = getattr(tree.body[0], "value", None)
        if expr is None or not isinstance(expr, ast.Call):
            return errors

        target = self._dotted_name(expr.func)
        if target is None or not self._is_supported_pennylane_method(
            target, method_prefixes
        ):
            return errors

        provided_args = {}
        # handle positional arguments
        for i, arg in enumerate(expr.args):
            if i < len(expected_args):
                provided_args[expected_args[i]["name"]] = arg

        # handle keyword arguments
        for keyword in expr.keywords:
            provided_args[keyword.arg] = keyword.value

        # check unexpected arguments
        for arg in provided_args.keys():
            if arg not in expected_args_names:
                errors.append(f"Unexpected argument '{arg}'")

        # check arguments existence and types
        for expected_arg in required_args:
            arg_name = expected_arg["name"]
            arg_type = expected_arg["type"]
            arg_description = expected_arg["description"]
            if arg_name not in provided_args:
                errors.append(
                    f"Missing required argument '{arg_name}'.\n{arg_name} ({arg_type}): {arg_description}"
                )

        return errors


class CodeValidator:
    """Main validator that orchestrates different validation strategies.

    This class coordinates validation by delegating to specialized validators
    for different aspects of code validation.

    Args:
        code (str): The generated code to validate

    Methods:
        validate_all: Validates the code using all available validation strategies
    """

    def __init__(self, code: str) -> None:
        self.source_code = code

        # Initialize validators
        self.syntax_validator = SyntaxValidator()
        self.doc_validator = PennyLaneDocValidator(
            reference_path=exp_const.PENNYLANE_REFERENCE_PATH
        )

    def validate_all(self) -> tuple[bool, list[str], float]:
        """Validate code using all available validation strategies."""
        overall_result = ValidationResult(is_valid=True, errors=[])

        # Syntax validation
        syntax_result = self.syntax_validator.validate(self.source_code)
        overall_result.merge(syntax_result)

        # PennyLane documentation validation
        doc_result = self.doc_validator.validate(self.source_code)
        overall_result.merge(doc_result)

        return (
            overall_result.is_valid,
            overall_result.errors,
            overall_result.cost,
        )


def is_same_code(last_code: str, code: str) -> bool:
    """check if last code is same as current code

    Args:
        last_code (str): generated code from last iteration
        code (str): generated code from current iteration

    Returns:
        bool: True if last code is same as current code
    """
    return last_code == code


@traceable(tags=exp_const.LANGSMITH_VALIDATION_CODE_TAGS)
def dry_run(
    experiment: qxmt.Experiment,
    config: dict,
    config_path: str,
) -> tuple[str, list[str]]:
    """Dry run the generated code by QXMT experiment.

    Args:
        experiment (qxmt.Experiment): qxmt experiment instance
        config (dict): config for dry run
        config_path (str): path to config file for dry run (used for temp file location)

    Returns:
        str: error message if code execution is failed
    """
    # save config in tmp file
    tmp_config_path = f"{config_path}".replace(".yaml", "_tmp.yaml")
    with open(tmp_config_path, "w") as file:
        yaml.dump(config, file)

    error_massage = ""
    warning_messages = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("default")
        try:
            experiment.run(config_source=tmp_config_path, add_results=False)
            os.remove(tmp_config_path)
            logger.info("dry run: Code execution is successful.")
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.warning(f"Error occurred during dry run: {e}")
            error_massage = _summarize_error_messages(
                e, target_file_name=DRYRUN_CODE_PATH
            )

        for warning in w:
            warning_messages.append(str(warning.message))

    return error_massage, warning_messages


def _summarize_error_messages(
    e: Exception, target_file_name: str | None = None
) -> str:
    """Summarize error messages from an exception. If target_file_name is provided and
    exists in the exception, it will only return the error messages from the target file.
    Otherwise, it will return the error messages from all files.

    Args:
        e (Exception): Exception to summarize
        target_file_name (str | None, optional): Target file name. Defaults to None.

    Returns:
        str: Summarized error messages
    """
    if target_file_name is None:
        return format_exc()

    te = TracebackException.from_exception(e)
    exc_msg = "".join(te.format_exception_only()).strip()

    # extract target file frame
    matching_frames = [
        frame for frame in te.stack if target_file_name in frame.filename
    ]

    if not matching_frames:
        return format_exc()

    lines = [exc_msg]
    for frame in matching_frames:
        file_path = frame.filename
        lineno = frame.lineno or 0
        func_name = frame.name
        try:
            src_line = (
                Path(file_path).read_text().splitlines()[lineno - 1].strip()
            )
        except Exception:
            src_line = "<source unavailable>"
        lines.append(
            f"File: {file_path}, Line: {lineno}, Function: {func_name}\nSource Code: {src_line}"
        )

    return "\n".join(lines)


class CodeValidationOrchestrator:
    """Orchestrates the complete code validation and regeneration process.

    This class breaks down the complex validation workflow into smaller,
    focused methods for better maintainability and testability.

    Args:
        llm_client (ChatClient): Client for interacting with the language model
        db_client (PineconeClient | None): Client for Pinecone database
        context (RunContext): Context of the current run
        experiment (qxmt.Experiment): QXMT experiment object for dry run
        dry_run_config_path (str): Path to dry run config
        max_retry (int): Maximum number of retry attempts
    """

    def __init__(
        self,
        llm_client: ChatClient,
        db_client: PineconeClient | None,
        context: RunContext,
        experiment: qxmt.Experiment,
        dry_run_config_path: str,
        max_retry: int = exp_const.DEFAULT_MAX_RETRY,
    ) -> None:
        self.llm_client = llm_client
        self.db_client = db_client
        self.context = context
        self.experiment = experiment
        self.dry_run_config_path = dry_run_config_path
        self.max_retry = max_retry
        self.code_generator = GenerateCode(
            client=llm_client,
            code_model_version=context.model_versions.code,
            reference_model_version=context.model_versions.reference,
            reasoning_effort=CodeValidationParams().reasoning_effort,
            verbosity=CodeValidationParams().verbosity,
        )

    @traceable(tags=exp_const.LANGSMITH_VALIDATION_CODE_TAGS)
    def validate_and_retry(
        self, generate_result: GeneratedResult, config_list: list[dict]
    ) -> tuple[GeneratedResult, float]:
        """Main orchestration method for code validation and regeneration.

        The validation process includes:
        1. Initial validation (syntax and documentation)
        2. Code regeneration if validation fails
        3. Code uniqueness check against previous code
        4. Dry run execution with retry

        Args:
            generate_result (GeneratedResult): Generated code result to validate
            config_list (list[dict]): List of dry run configs

        Returns:
            tuple[GeneratedResult, float]: Validated code result and total validation cost

        Raises:
            ValueError: If valid code cannot be generated after max_retry attempts
        """
        validation_cost = 0.0

        # Phase 1: Initial validation and regeneration
        validated_result, initial_cost = (
            self._run_initial_validation_with_retry(generate_result)
        )
        validation_cost += initial_cost

        # Phase 2: Code uniqueness check
        if self._is_same_as_previous_code(validated_result):
            logger.info("Generated code is same as last code.")
            return validated_result, validation_cost

        # Phase 3: Dry run with retry
        final_result, dry_run_cost = self._perform_dry_run_with_retry(
            validated_result, config_list
        )
        validation_cost += dry_run_cost

        return final_result, validation_cost

    def _run_initial_validation_with_retry(
        self, generate_result: GeneratedResult
    ) -> tuple[GeneratedResult, float]:
        """Run initial validation and regenerate code if validation fails.

        Args:
            generate_result (GeneratedResult): Generated code result to validate

        Returns:
            tuple[GeneratedResult, float]: Validated result and cost

        Raises:
            ValueError: If valid code cannot be generated after max_retry attempts
        """
        re_generate_code_result = copy.deepcopy(generate_result.implement)
        validation_cost = 0.0
        retry_num = 0
        validation_message_history: MESSAGE_HISTORY_TYPE = []

        while True:
            validation_result = self._run_validation(
                re_generate_code_result.code
            )
            validation_cost += validation_result.cost

            if validation_result.is_valid:
                return (
                    GeneratedResult(
                        idea=generate_result.idea,
                        score=generate_result.score,
                        implement=re_generate_code_result,
                    ),
                    validation_cost,
                )

            # Handle validation failure
            logger.info("Generated code is invalid. Retry to generate code...")
            re_generate_code_result, validation_message_history, cost = (
                self._regenerate_code(
                    code=re_generate_code_result.code,
                    error_messages=validation_result.errors,
                    warning_messages=[],
                    message_history=validation_message_history,
                )
            )
            validation_cost += cost
            retry_num += 1

            if retry_num > self.max_retry:
                raise ValueError("Failed to generate valid code.")

    def _run_validation(self, code: str) -> ValidationResult:
        """Run validation on the provided code.

        Args:
            code (str): Code to validate

        Returns:
            ValidationResult: Validation result with errors and cost
        """
        validator = CodeValidator(code=code)
        is_valid, error_messages, cost = validator.validate_all()

        return ValidationResult(
            is_valid=is_valid, errors=error_messages, cost=cost
        )

    def _get_regenerate_tools(self) -> list[dict[str, str]] | None:
        if "codex" in self.code_generator.model_version:
            return None
        else:
            return [
                {"type": "web_search_preview"},
                {"type": "code_interpreter", "container": {"type": "auto"}},  # type: ignore
            ]

    def _regenerate_code(
        self,
        code: str,
        error_messages: list[str],
        warning_messages: list[str],
        message_history: MESSAGE_HISTORY_TYPE,
    ) -> tuple:
        """Regenerate code based on validation errors.

        Args:
            code (str): Current code that failed validation
            error_messages (list[str]): Validation error messages
            warning_messages (list[str]): Warning messages
            message_history (MESSAGE_HISTORY_TYPE): Previous message history

        Returns:
            tuple: New generated code result, updated message history, and cost
        """
        developer_prompt, user_prompt = FixCodePrompt(
            code=code,
            error_messages=error_messages,
            warning_messages=warning_messages,
        ).build()

        return self.code_generator.generate(
            developer_prompt=developer_prompt,
            user_prompt=user_prompt,
            message_history=message_history,
            n_history=None,
            tools=self._get_regenerate_tools(),
        )

    def _is_same_as_previous_code(
        self, validated_result: GeneratedResult
    ) -> bool:
        """Check if generated code is same as previous code.

        Args:
            validated_result (GeneratedResult): Validated code result

        Returns:
            bool: True if code is same as previous code
        """
        return is_same_code(
            last_code=self.context.last_code,
            code=validated_result.implement.code,
        )

    def _perform_dry_run_with_retry(
        self, validate_result: GeneratedResult, config_list: list[dict]
    ) -> tuple[GeneratedResult, float]:
        """Perform dry run execution with retry on failure.

        Args:
            validate_result (GeneratedResult): Validated code result
            config_list (list[dict]): List of dry run configs

        Returns:
            tuple[GeneratedResult, float]: Final result after dry run and cost
        """
        current_result = validate_result
        validation_cost = 0.0
        validation_message_history: MESSAGE_HISTORY_TYPE = []

        for _ in range(self.max_retry):
            all_error_messages = []
            all_warning_messages = []
            # Save code for dry run
            save_generated_code(
                f"{exp_const.GENERATED_CODE_DIRC}/{self.experiment.name}/{DRYRUN_CODE_PATH}",
                current_result.implement.code,
            )

            for i, config in enumerate(config_list):
                logger.info(f"Executing dry run {i + 1}/{len(config_list)}")
                # Execute dry run
                error_message, warning_messages = dry_run(
                    experiment=self.experiment,
                    config=config,
                    config_path=self.dry_run_config_path,
                )
                if error_message and error_message.strip():
                    all_error_messages.append(error_message)

                all_warning_messages += warning_messages

            if len(all_error_messages) == 0:
                # Dry run successful
                break
            else:
                # Regenerate code based on dry run errors
                logger.info("Dry run failed. Regenerating code...")
                re_generate_code_result, validation_message_history, cost = (
                    self._regenerate_code(
                        code=current_result.implement.code,
                        error_messages=all_error_messages,
                        warning_messages=all_warning_messages,
                        message_history=validation_message_history,
                    )
                )
                validation_cost += cost

                current_result = GeneratedResult(
                    idea=validate_result.idea,
                    score=validate_result.score,
                    implement=re_generate_code_result,
                )

        return current_result, validation_cost
