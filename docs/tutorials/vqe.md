# VQE Experiment Tutorial

This tutorial guides you through setting up and running a Variational Quantum Eigensolver (VQE) experiment using **Astronaut**. The agent will autonomously design ansatzes (quantum circuits) to estimate the ground state energy of a given Hamiltonian.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Running the Experiment](#running-the-experiment)
4. [Understanding the Output](#understanding-the-output)

---

## Prerequisites

Ensure you have completed the following steps:

1.  **Installed dependencies**: See [Installation guide](../../README.md#installation).
2.  **Configured Environment Variables**: API keys in `.env` are required.
3.  **Prepared External Knowledge (Optional but Recommended)**:
    -   **PennyLane Documentation**: For code validation and reference.
    -   **ArXiv Papers**: For research-based idea generation and brushing up.
    -   *See [Database Setup](./docs/tutorials/database_setup.md) for instructions.*

---

## Configuration

There are three main configuration files for running a VQE experiment:

1.  **Experiment Configuration** (`configs/vqe/experiment.yaml`): Controls the agent's behavior (e.g., number of trials, research mode).
2.  **Dataset Configuration** (`configs/vqe/datasets.yaml`): Defines the molecule settings to be used.
3.  **QXMT Experiment Management Configuration** (`configs/vqe/base/astronaut.yaml`): Manages the underlying QXMT settings (e.g., quantum device, optimizer settings).

> **Note**: The **Dataset Configuration** and **QXMT Experiment Management Configuration** rely on the external library **[QXMT](https://github.com/Qyusu/qxmt)**. Please refer to its documentation for detailed setup.

### 1. Experiment Configuration
The experiment is controlled by a YAML configuration file. The default configuration is located at `configs/vqe/experiment.yaml`.

### Key Parameters

| Parameter | Type | Description | Recommendation for Beginners |
| :--- | :--- | :--- | :--- |
| `task_type` | `string` | Must be set to `"vqe"`. | Do not change. |
| `dataset_config_path` | `string` | Path to the dataset configuration file (relative to the experiment config). | Keep as `"datasets.yaml"`. |
| `validation_qubits` | `list[int]` | Qubit counts used to validate the scalability of the generated circuit. | Set the list of qubits counts for molecules defined in datasets.yaml. |
| `experiment_name` | `string` | A unique identifier for your experiment run. Output folders will use this name. | Use a descriptive name (e.g., `vqe_tutorial_run`). |
| `max_trial_num` | `integer` | Maximum number of iterations the agent will perform. | Start with `5` to test the waters. |
| `max_idea_num` | `integer` | Maximum number of ideas generated simultaneously per trial. | `2` is a good balance. Note that increasing this value will linearly increase API costs. |
| `max_suggestion_num` | `integer` | Maximum number of improvement suggestions during idea review. | `2` is usually sufficient. |
| `expanding_steps` | `list[int]` | Number of idea in each expanding step. | `[5, 3]` is a good balance. |
| `research_mode` | `string` | Depth of initial research (`"deep"`, `"light"`, `"none"`). | Use `"light"` initially to save tokens and time, or set an existing report via `existing_research_report_path`. |
| `existing_research_report_path` | `string` | Path to an existing or original research report. | Set to `null` if no existing report exists. |
| `reflection_type` | `string` | Strategy for refining ideas (`"debate"`, `"self"`, `"none"`). | `"self"` is faster; `"debate"` is more thorough. |
| `max_reflection_round` | `integer` | Maximum number of reflection rounds. | `2` is usually sufficient. |
| `max_scoring_round` | `integer` | Maximum number of scoring rounds. If set to `0`, scoring is disabled. | `0` is usually sufficient. If quantitative evaluation of ideas is required, set to `1`. |
| `log_level` | `string` | Logging level (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`). The log file saved in `logs/<experiment_name>/` will be at this level. | `"DEBUG"` is usually sufficient. |
| `console_log_level` | `string` | Console logging level (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`). | `"INFO"` is usually sufficient. |


<details><summary>Example Configuration</summary>

```yaml
task_type: "vqe"
dataset_config_path: "datasets.yaml"
validation_qubits: [4, 6, 8, 10, 12]
experiment_name: "vqe_tutorial_run"
desc: "This is a VQE experiment."
max_trial_num: 5
max_idea_num: 2
max_suggestion_num: 2
expanding_steps: [5, 3]
research_mode: "light"
existing_research_report_path: null
reflection_type: "debate"
max_reflection_round: 2
max_scoring_round: 0
log_level: "DEBUG"
console_log_level: "INFO"
```
</details>

### 2. Dataset Configuration
The molecule settings are controlled by a YAML configuration file. The default configuration is located at `configs/vqe/datasets.yaml`.

<details><summary>Example Configuration</summary>

```yaml
H2:
  name: "H2"
  symbols: ["H", "H"]
  n_qubits: 4
  structures:
    charge: 0
    multi: 1
    active_electrons: 2
    active_orbitals: 2
  hamiltonian_settings:
    basis_name: "STO-3G"
    method: "pyscf"
    unit: "angstrom"
  bond_length_settings:
    min_bond_length: 0.2
    max_bond_length: 2.5
    sample_nums: 10
    equilibrium_bond_length: 0.741
  reference_energy_methods: ["casci", "casscf", "fci"]
  exact_energy: "fci_energy"
  energy_cache:
    0.2:
      hf: 0.16417501206169538
      casci: 0.15748213479836348
      casscf: 0.15748213479836348
      fci: 0.15748213479836348
    0.4875:
      hf: -1.0312097366725492
      casci: -1.0430438122856938
      casscf: -1.0430438122856933
      fci: -1.0430438122856938
```
</details>

### 3. QXMT Experiment Management Configuration
The experiment is controlled by a YAML configuration file. The default configuration is located at `configs/vqe/base/astronaut.yaml`. Details of the QXMT configuration about VQE can be found in [QXMT VQE Tutorial](https://qyusu.github.io/qxmt/tutorials/en/vqe_tutorial.html).

#### Key Parameters
| Parameter | Type | Description | Recommendation for Beginners |
| :--- | :--- | :--- | :--- |
| `hamiltonian` | `dict` | Molecular Hamiltonian settings. | Automatically configure settings for molecules defined in `configs/vqe/datasets.yaml`. with in the experiment.|
| `device.device_name` | `string` | Your quantum device name. Details in [About Device](https://qyusu.github.io/qxmt/tutorials/en/tool_reference.html#device). | `"lightning.qubit"` is faster. |
| `ansatz` | `dict` | VQE ansatz settings. | Automatically configure settings for generated ansatz in experiment. |
| `model` | `dict` | VQE model settings. | See [QXMT VQE Tutorial](https://qyusu.github.io/qxmt/tutorials/en/vqe_tutorial.html#vqe-configuration) for details. |
| `evaluation.default_metrics` | `list[string]` | Your evaluation metrics (e.g., `"final_cost"`, `"hf_energy"`, `"casci_energy"`, `"casscf_energy"`, `"fci_energy"`). | select your metrics. |

<details><summary>Example Configuration</summary>

```yaml
description: "This is a template configuration file for H6 molecule."

global_settings:
  random_seed: &global_seed 42
  model_type: "vqe"

# The Hamiltonian configuration is dynamically updated
# at runtime based on each molecule’s configuration.
# Details defined in `configs/vqe/datasets.yaml`
hamiltonian:
  module_name: "qxmt.hamiltonians.pennylane"
  implement_name: "MolecularHamiltonian"
  params: null

device:
  platform: "pennylane"
  device_name: "lightning.qubit"
  n_qubits: 4
  shots: null
  random_seed: *global_seed

# The ansatz configuration is overwritten at runtime.
# So you don't need to change this setting.
ansatz:
  module_name: "astronaut.seed.vqe.ansatz"
  implement_name: "BaseAnsatz"
  params: null

model:
  name: "basic"
  diff_method: "best"
  optimizer_settings:
    name: "scipy.SLSQP"
    params:
      options:
        ftol: 1e-8
        disp: true
  params:
    max_steps: 2000
    init_params:
      type: "random"
      min_value: 0.0
      max_value: 0.1
    verbose: false

evaluation:
  default_metrics:
  - "final_cost"
  - "hf_energy"
  - "casci_energy"
  - "casscf_energy"
  - "fci_energy"
  custom_metrics: []
```
</details>

---

## Running the Experiment

1.  Move to the `astronaut` source directory:
    ```bash
    cd astronaut
    ```

2.  Launch the VQE agent:
    ```bash
    python3 main.py "../configs/vqe/experiment.yaml"
    ```

    The agent will begin by formulating a research plan, generating ansatz hypotheses, and then writing and testing PennyLane code.

---

## Understanding the Output
### Directory Structure
-   **`astronaut/generated/<experiment_name>/`**: Generated Python code for the quantum ansatz.
-   **`experiments/<experiment_name>/`**: QXMT files about settings and results.
-   **`configs/generated/<experiment_name>/`**: QXMT configuration file for each experiment.
-   **`logs/<experiment_name>/`**: Detailed logs of generated ideas and the agent's reasoning.


### Success Criteria
For VQE experiments, the primary metric is the **Energy Error** (difference between the estimated ground state energy and the exact value). Lower is better. Astronaut tries to minimize this error by iteratively refining the ansatz structure.
