from pathlib import Path
from typing import Final

MODULE_HOME: Final[Path] = Path(__file__).resolve().parents[1]
MODULE_SRC: Final[Path] = Path(__file__).resolve().parents[0]
CURRENT_WORKING_DIR: Final[Path] = Path.cwd()

PENNYLANE_VERSION: Final[str] = "0.39.0"
PENNYLANE_METHOD_PREFIXES: Final[list[str]] = ["qml", "qml.qchem"]

TASK_TYPE_PLACEHOLDER: Final[str] = "--TASK_TYPE_PLACEHOLDER--"
QKERNEL_MODEL_TYPE: Final[str] = "quantum_kernel"
VQE_MODEL_TYPE: Final[str] = "vqe"

CONFIG_DIRC: Final[str] = f"{MODULE_HOME}/configs"
PAPER_PDF_DIRC: Final[str] = f"{MODULE_HOME}/data/papers"
PENNYLANE_REFERENCE_PATH: Final[str] = (
    f"{MODULE_HOME}/data/references/v{PENNYLANE_VERSION}.json"
)
BASE_CONFIG_PATH: Final[str] = (
    f"{CONFIG_DIRC}/{TASK_TYPE_PLACEHOLDER}/base/astronaut.yaml"
)
DRYRUN_CONFIG_PATH: Final[str] = (
    f"{CONFIG_DIRC}/{TASK_TYPE_PLACEHOLDER}/base/dry_run.yaml"
)
GENERATED_CONFIG_DIRC: Final[str] = f"{CONFIG_DIRC}/generated"
QKERNEL_SEED_CODE_PATH: Final[str] = (
    f"{MODULE_SRC}/seed/{QKERNEL_MODEL_TYPE}/feature_map.py"
)
VQE_SEED_CODE_PATH: Final[str] = f"{MODULE_SRC}/seed/{VQE_MODEL_TYPE}/ansatz.py"
GENERATED_CODE_DIRC: Final[str] = f"{MODULE_SRC}/generated"
GENERATED_MODULE_ROOT: Final[str] = "astronaut.generated"
RESEARCH_REPORT_FILE_NAME: Final[str] = "research_report.md"

# default values of max number of trials
DEFAULT_MAX_TRIAL_NUM: Final[int] = 10

# default values of max retry number when error occurs
DEFAULT_MAX_RETRY: Final[int] = 3

# default value of max number of ideas for generation component
DEFAULT_MAX_IDEA_NUM: Final[int] = 2

# default value of max number of suggestions when reviewing the idea
DEFAULT_MAX_SUGGESTION_NUM: Final[int] = 3

# default value of number of ideas for each steps of expanding
DEFAULT_EXPANDING_STEPS: Final[list[int]] = [10, 5]

# default value of max round of reflection when idea refinement
DEFAULT_MAX_REFLECTION_ROUND: Final[int] = 2

# default value of max number of scoring rounds
DEFAULT_MAX_SCORING_ROUND: Final[int] = 0

# Special Strings
COMPLETED: Final[str] = "COMPLETED"
NOT_PROVIDED_INFORMATION: Final[str] = "NOT PROVIDED"

# Tracer Tags
LANGSMITH_EXPERIMENT_TAGS: Final[list[str]] = ["experiment"]
LANGSMITH_RETRIEVAL_ARXIV_TAGS: Final[list[str]] = ["retrieval", "arxiv"]
LANGSMITH_GENERATION_TAGS: Final[list[str]] = ["generation"]
LANGSMITH_SCORING_IDEA_TAGS: Final[list[str]] = ["scoring", "idea"]
LANGSMITH_DEEP_RESEARCH_IDEA_TAGS: Final[list[str]] = ["deep-research", "idea"]
LANGSMITH_SUMMARY_PAPER_TAGS: Final[list[str]] = ["summary", "paper"]
LANGSMITH_PENNYLANE_REFERENCE_TAGS: Final[list[str]] = ["pennylane_reference"]
LANGSMITH_VALIDATION_SYNTAX_TAGS: Final[list[str]] = ["validation", "syntax"]
LANGSMITH_VALIDATION_CODE_TAGS: Final[list[str]] = ["validation", "code"]
LANGSMITH_REVIEW_IDEA_TAGS: Final[list[str]] = ["review", "idea"]
LANGSMITH_SELF_REFLECTION_IDEA_TAGS: Final[list[str]] = [
    "self-reflection",
    "idea",
]
LANGSMITH_DEBATE_REFLECTION_IDEA_TAGS: Final[list[str]] = [
    "debate-reflection",
    "idea",
]
LANGSMITH_CRITIC_IDEA_TAGS: Final[list[str]] = ["critic", "idea"]
LANGSMITH_ADVOCATE_IDEA_TAGS: Final[list[str]] = ["advocate", "idea"]
LANGSMITH_EXPERT_PAPER_TAGS: Final[list[str]] = ["expert", "paper"]
LANGSMITH_GENERATION_IDEA_TAGS: Final[list[str]] = ["generation", "idea"]
LANGSMITH_GENERATION_CODE_TAGS: Final[list[str]] = ["generation", "code"]
LANGSMITH_EXPANDING_TAGS: Final[list[str]] = ["expanding"]
