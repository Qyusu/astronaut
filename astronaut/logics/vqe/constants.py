from typing import Final

# General IDs
IDEA_ID_COL: Final[str] = "idea_id"
MOLECULE_NAME_COL: Final[str] = "molecule_name"
BOND_LENGTH_COL: Final[str] = "bond_length"
COST_COL: Final[str] = "cost"
NUM_GATES_ORDER_COL: Final[str] = "num_gates_order"
NUM_PARAMS_ORDER_COL: Final[str] = "num_params_order"
QXMT_FINAL_COST_COL: Final[str] = "final_cost"
AGGREGATED_COST_COL: Final[str] = "aggregated_cost"
AGGREGATED_RUN_TIME_COL: Final[str] = "aggregated_run_time"
AGGREGATED_NUM_ITER_COL: Final[str] = "aggregated_num_iter"

# QXMT Config Keys
QXMT_CONFIG_KEY_DESCRIPTION: Final[str] = "description"
QXMT_CONFIG_KEY_ANSATZ: Final[str] = "ansatz"
QXMT_CONFIG_KEY_MODULE_NAME: Final[str] = "module_name"
QXMT_CONFIG_KEY_IMPLEMENT_NAME: Final[str] = "implement_name"
QXMT_CONFIG_KEY_MODEL: Final[str] = "model"
QXMT_CONFIG_KEY_PARAMS: Final[str] = "params"
QXMT_CONFIG_KEY_INIT_PARAMS: Final[str] = "init_params"
QXMT_CONFIG_KEY_HAMILTONIAN: Final[str] = "hamiltonian"
QXMT_CONFIG_KEY_DEVICE: Final[str] = "device"
QXMT_CONFIG_KEY_N_QUBITS: Final[str] = "n_qubits"
QXMT_CONFIG_KEY_SYMBOLS: Final[str] = "symbols"
QXMT_CONFIG_KEY_COORDINATES: Final[str] = "coordinates"

# Init Params Keys
INIT_PARAMS_TYPE: Final[str] = "type"
INIT_PARAMS_VALUES: Final[str] = "values"
INIT_PARAMS_TYPE_CUSTOM: Final[str] = "custom"

# Prefixes
SEED_PHASE_ID_PREFIX: Final[str] = "seed"
EXPANDING_PHASE_ID_PREFIX: Final[str] = "expanding_step"
CONFIG_FILE_PREFIX: Final[str] = "config"
DRYRUN_FILE_PREFIX: Final[str] = "dry_run"
CODE_FILE_PREFIX: Final[str] = "ansatz"
