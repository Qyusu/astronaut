from pathlib import Path

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODULE_SRC: Path = Path(__file__).resolve().parents[0]
CURRENT_WORKING_DIR: Path = Path.cwd()

CONFIG_DIRC = f"{MODULE_HOME}/configs"
PAPER_PDF_DIRC = f"{MODULE_HOME}/data/papers"
BASE_CONFIG_PATH = f"{CONFIG_DIRC}/base/astronaut.yaml"
DRYRUN_CONFIG_PATH = f"{CONFIG_DIRC}/base/dry_run.yaml"
GENERATED_CONFIG_DIRC = f"{CONFIG_DIRC}/generated"
SEED_CODE_PATH = f"{MODULE_SRC}/seed/feature_map.py"
GENERATED_CODE_DIRC = f"{MODULE_SRC}/generated"
GENERATED_MODULE_ROOT = "astronaut.generated"

# default values of max number of trials
DEFAULT_MAX_TRIAL_NUM = 10

# default values of max retry number when error occurs
DEFAULT_MAX_RETRY = 3

# default value of max number of ideas for generation component
DEFAULT_MAX_IDEA_NUM = 2

# default value of max number of suggestions when reviewing the idea
DEFAULT_MAX_SUGGESTION_NUM = 3

# default value of max round of reflection when idea refinement
DEFAULT_MAX_REFLECTION_ROUND = 3

# OpenAI Model Series
STABLE_REASONING_MODEL_VERSIONS = ["o3-mini-2025-01-31", "o1-pro-2025-03-19", "o1-2024-12-17"]
REASONING_SERIES = [
    "o1-mini",
    "o1-mini-2024-09-12",
    "o1-preview",
    "o1-preview-2024-09-12",
    "o1",
    "o1-2024-12-17",
    "o1-pro-2025-03-19",
    "o3-mini",
    "o3-mini-2025-01-31",
]

GPT_SERIES = [
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-11-20",
    "gpt-4.5-preview",
    "gpt-4.5-preview-2025-02-27",
]

REASONING_MAX_TOKENS = 100000
# O1_MAX_TOKENS = 32768
GPT_MAX_TOKENS = 16384

# Anthropic Model Series
ANTHOROPIC_THINKING_SERIES = ["claude-3-7-sonnet", "claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219"]

# Pennylane Package Version
PENNYLANE_VERSION = "0.39.0"

# Special Strings
COMPLETED = "COMPLETED"
