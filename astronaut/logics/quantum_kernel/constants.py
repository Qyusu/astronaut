from typing import Final

# General IDs
IDEA_ID_COL: Final[str] = "idea_id"
DATASET_NAME_COL: Final[str] = "dataset_name"
RUN_TIME_COL: Final[str] = "run_time"
AGGREGATED_RUN_TIME_COL: Final[str] = "aggregated_run_time"

# Config Keys
QXMT_CONFIG_KEY_DESCRIPTION: Final[str] = "description"
QXMT_CONFIG_KEY_DATASET: Final[str] = "dataset"
QXMT_CONFIG_KEY_FEATURE_MAP: Final[str] = "feature_map"
QXMT_CONFIG_KEY_MODEL: Final[str] = "model"
QXMT_CONFIG_KEY_PARAMS: Final[str] = "params"
QXMT_CONFIG_KEY_MODULE_NAME: Final[str] = "module_name"
QXMT_CONFIG_KEY_IMPLEMENT_NAME: Final[str] = "implement_name"
QXMT_CONFIG_KEY_DEVICE: Final[str] = "device"
QXMT_CONFIG_KEY_N_QUBITS: Final[str] = "n_qubits"
QXMT_CONFIG_KEY_RAW_PREPROCESS_LOGIC: Final[str] = "raw_preprocess_logic"
QXMT_CONFIG_KEY_N_SAMPLES: Final[str] = "n_samples"
QXMT_CONFIG_KEY_EVALUATION: Final[str] = "evaluation"
QXMT_CONFIG_KEY_DEFAULT_METRICS: Final[str] = "default_metrics"

# Prefixes
SEED_PHASE_ID_PREFIX: Final[str] = "seed"
EXPANDING_PHASE_ID_PREFIX: Final[str] = "expanding_step"
CODE_FILE_PREFIX: Final[str] = "feature_map"
CONFIG_FILE_PREFIX: Final[str] = "config"
DRYRUN_FILE_PREFIX: Final[str] = "dry_run"
AGGREGATED_COL_PREFIX: Final[str] = "aggregated"

# Suffixes
VALIDATION_SUFFIX: Final[str] = "validation"
TEST_SUFFIX: Final[str] = "test"
HP_CONFIG_SUFFIX: Final[str] = "search"

# Metrics
METRIC_OPTIMIZATION_DIRECTIONS: Final[dict[str, str]] = {
    "accuracy": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "f1_score": "maximize",
    "mean_absolute_error": "minimize",
    "root_mean_squared_error": "minimize",
    "r2_score": "maximize",
}

# Hyperparameter Search Config
HP_SEARCH_N_SAMPLES: Final[int] = 10000
HP_SEARCH_SPACE: Final[dict[str, list[float]]] = {"C": [0.001, 100.0]}
HP_SEARCH_CV: Final[int] = 5
HP_SEARCH_N_JOBS: Final[int] = 3
HP_SEARCH_N_TRIALS: Final[int] = 500
HP_SAMPLER_TYPE: Final[str] = "tpe"
