from pathlib import Path

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODULE_SRC: Path = Path(__file__).resolve().parents[0]
CURRENT_WORKING_DIR: Path = Path.cwd()

PENNYLANE_VERSION = "0.39.0"

MODEL_TYPE_PLACEHOLDER = "--MODEL_TYPE_PLACEHOLDER--"
QKERNEL_MODEL_TYPE = "quantum_kernel"
VQE_MODEL_TYPE = "vqe"

CONFIG_DIRC = f"{MODULE_HOME}/configs"
PAPER_PDF_DIRC = f"{MODULE_HOME}/data/papers"
PENNYLANE_REFERENCE_PATH = (
    f"{MODULE_HOME}/data/references/v{PENNYLANE_VERSION}.json"
)
BASE_CONFIG_PATH = f"{CONFIG_DIRC}/base/{MODEL_TYPE_PLACEHOLDER}/astronaut.yaml"
DRYRUN_CONFIG_PATH = f"{CONFIG_DIRC}/base/{MODEL_TYPE_PLACEHOLDER}/dry_run.yaml"
GENERATED_CONFIG_DIRC = f"{CONFIG_DIRC}/generated"
QKERNEL_SEED_CODE_PATH = (
    f"{MODULE_SRC}/seed/{QKERNEL_MODEL_TYPE}/feature_map.py"
)
VQE_SEED_CODE_PATH = f"{MODULE_SRC}/seed/{VQE_MODEL_TYPE}/ansatz.py"
GENERATED_CODE_DIRC = f"{MODULE_SRC}/generated"
GENERATED_MODULE_ROOT = "astronaut.generated"
RESEARCH_REPORT_FILE_NAME = "research_report.md"

# default values of max number of trials
DEFAULT_MAX_TRIAL_NUM = 10

# default values of max retry number when error occurs
DEFAULT_MAX_RETRY = 3

# default value of max number of ideas for generation component
DEFAULT_MAX_IDEA_NUM = 2

# default value of max number of suggestions when reviewing the idea
DEFAULT_MAX_SUGGESTION_NUM = 3

# default value of number of ideas for each steps of filtering
DEFAULT_FILTERING_STEPS = [10, 5]

# default value of max round of reflection when idea refinement
DEFAULT_MAX_REFLECTION_ROUND = 2

# default value of max number of scoring rounds
DEFAULT_MAX_SCORING_ROUND = 0

# Anthropic Model Series
ANTHOROPIC_THINKING_SERIES = [
    "claude-3-7-sonnet",
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-20250219",
]

# Special Strings
COMPLETED = "COMPLETED"
NOT_PROVIDED_INFORMATION = "NOT PROVIDED"
