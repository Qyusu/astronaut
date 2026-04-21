# Quantum Kernel Experiment Tutorial

This tutorial guides you through setting up and running a Quantum Kernel experiment using **Astronaut**. The goal of this experiment is to automatically design and optimize quantum feature maps for kernel-based classification and regression tasks.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Running the Experiment](#running-the-experiment)
4. [Understanding the Output](#understanding-the-output)

---

## Prerequisites

Before running any experiment, ensure you have:

1.  **Installed dependencies**: Follow the [Installation guide](../../README.md#installation).
2.  **Configured Environment Variables**: Set up your `.env` file with necessary API keys (OpenAI, etc.).
3.  **Prepared External Knowledge (Optional but Recommended)**:
    -   **PennyLane Documentation**: For code validation and reference.
    -   **ArXiv Papers**: For research-based idea generation and brushing up.
    -   *See [Database Setup](./docs/tutorials/database_setup.md) for instructions.*

---

## Configuration

There are three main configuration files for running a Quantum Kernel experiment:

1.  **Experiment Configuration** (`configs/quantum_kernel/experiment.yaml`): Controls the agent's behavior (e.g., number of trials, research mode).
2.  **Dataset Configuration** (`configs/quantum_kernel/datasets.yaml`): Defines the dataset to be used (e.g., path, preprocessing).
3.  **QXMT Experiment Management Configuration** (`configs/quantum_kernel/base/astronaut.yaml`): Manages the underlying QXMT settings (e.g., quantum device, kernel type).

> **Note**: The **Dataset Configuration** and **QXMT Experiment Management Configuration** rely on the external library **[QXMT](https://github.com/Qyusu/qxmt)**. Please refer to its documentation for detailed setup.


### 1. Experiment Configuration
The experiment is controlled by a YAML configuration file. The default configuration is located at `configs/quantum_kernel/experiment.yaml`.

#### Key Parameters

| Parameter | Type | Description | Recommendation for Beginners |
| :--- | :--- | :--- | :--- |
| `task_type` | `string` | Must be set to `"quantum_kernel"`. | Do not change. |
| `dataset_config_path` | `string` | Path to the dataset configuration file (relative to the experiment config). | Keep as `"datasets.yaml"`. |
| `validation_qubits` | `list[int]` | Qubit counts used to validate the scalability of the generated circuit. | `[2, 4, 8, 10]` for faster tests. |
| `experiment_name` | `string` | A unique identifier for your experiment run. Output folders will use this name. | Use a descriptive name (e.g., `qk_tutorial_run`). |
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
task_type: "quantum_kernel"
dataset_config_path: "datasets.yaml"
validation_qubits: [2, 4, 6, 8, 10]
experiment_name: "qk_tutorial_run"
desc: "This is a quantum kernel experiment."
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
The dataset is controlled by a YAML configuration file. The default configuration is located at `configs/quantum_kernel/datasets.yaml`. The details about QXMT dataset configuration can be found in [`Configuring Run Settings`](https://qyusu.github.io/qxmt/tutorials/en/qkernel_mnist_custom.html#configuring-run-settings) and [`About Dataset`](https://qyusu.github.io/qxmt/tutorials/en/tool_reference.html#dataset).

<details><summary>Example Configuration</summary>

#### MNIST, sampling and preprocessing by QXMT
```yaml
mnist:
  file:
    data_path: "data/datasets/mnist/X_train.npy"
    label_path: "data/datasets/mnist/y_train.npy"
    label_name: null
  split:
    train_ratio: 0.6
    validation_ratio: 0.2
    test_ratio: 0.2
    shuffle: true
  features: null
  raw_preprocess_logic:
    module_name: "qxmt.datasets.raw_preprocess.sampling"
    implement_name: "sampling_by_each_class"
    params:
      n_samples: 60000
      labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
      random_seed: 42
  transform_logic:
    module_name: "qxmt.datasets.transform.reducer"
    implement_name: "dimension_reduction_by_pca"
    params:
      n_components: 80
      normalize: true
      random_seed: 42
```

#### Simple setting for custom dataset
```yaml
my_dataset:
  file:
    data_path: "data/datasets/my_dataset.csv"
    label_path: null
    label_name: "target_col_name"
  split:
    train_ratio: 0.6
    validation_ratio: 0.2
    test_ratio: 0.2
    shuffle: true
  features: null
  raw_preprocess_logic: null
  transform_logic: null
```
</details>

### 3. QXMT Experiment Management Configuration
The experiment is controlled by a YAML configuration file. The default configuration is located at `configs/quantum_kernel/base/astronaut.yaml`.

#### Key Parameters
| Parameter | Type | Description | Recommendation for Beginners |
| :--- | :--- | :--- | :--- |
| `task_type` | `string` | Your quantum machine learning task type. Select from `classification` or `regression`. | select your task type. |
| `dataset` | `string` | Your dataset settings. | Automatically configure settings for molecules defined in `configs/quantum_kernel/datasets.yaml`. with in the experiment. |
| `device.device_name` | `string` | Your quantum device name. Details in [About Device](https://qyusu.github.io/qxmt/tutorials/en/tool_reference.html#device). | `"lightning.qubit"` is faster. |
| `feature_map` | `dict` | Feature map settings. | Automatically configure settings for generated feature map in experiment. |
| `kernel.implement_name` | `string` | Your quantum kernel implement name. Details in [About Kernel](https://qyusu.github.io/qxmt/tutorials/en/tool_reference.html#kernel). | `FidelityKernel` is usually good. |
| `model.name` | `string` | Your quantum machine learning model name. Select from `qsvc`, `qsvr` or `qrigge`. | regression is `qsvr` and classification is `qsvc`. |
| `evaluation.default_metrics` | `list[string]` | Your evaluation metrics. If regression task select from `mean_absolute_error`, `root_mean_squared_error`, `r2_score`. If classification task select from `accuracy`, `precision`, `recall`, `f1_score`. | select your metrics. |


<details><summary>Example Configuration</summary>

```yaml
description: "This is base settings for Astronaut experiment."

global_settings:
  random_seed: &global_seed 42
  model_type: "qkernel"
  task_type: "classification"

# The dataset configuration is overwritten at runtime.
# So you don't need to change this setting.
# Details defined in `configs/quantum_kernel/datasets.yaml`
dataset: null

device:
  platform: "pennylane"
  device_name: "lightning.qubit"
  n_qubits: 10
  shots: null

# The feature map configuration is overwritten at runtime.
# So you don't need to change this setting.
feature_map:
  module_name: "astronaut.seed.quantum_kernel.feature_map"
  implement_name: "SeedFeatureMap"
  params: {
    n_qubits: 10
  }

kernel:
  module_name: "qxmt.kernels.pennylane"
  implement_name: "FidelityKernel"
  params: {}

model:
  name: "qsvc"
  params:
    C: 1.0
    gamma:  "scale"

evaluation:
  default_metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1_score"
  custom_metrics: []

```
</details>


---

## Running the Experiment

1.  Navigate to the `astronaut` directory (where `main.py` is located):
    ```bash
    cd astronaut
    ```

2.  Run the experiment using the configuration file:
    ```bash
    python3 main.py "../configs/quantum_kernel/experiment.yaml"
    ```

    *Note: The path to the config file is relative to your current working directory.*

3.  **Monitor the Console**: You will see logs indicating the agent's progress:
    -   Researching literature...
    -   Generating ideas...
    -   Implementing code...
    -   Validating circuits...

---

## Understanding the Output
### Directory Structure
-   **`astronaut/generated/<experiment_name>/`**: Generated Python code for the quantum feature maps.
-   **`experiments/<experiment_name>/`**: QXMT files about settings and results.
-   **`configs/generated/<experiment_name>/`**: QXMT configuration file for each experiment.
-   **`logs/<experiment_name>/`**: Detailed logs of generated ideas and the agent's reasoning.

### Success Criteria
For Quantum Kernel experiments, the primary metric is the first value specified in `evaluation.default_metrics` (e.g., `accuracy` for classification or `mean_absolute_error` for regression). The optimization direction (maximization or minimization) is automatically determined based on the primary metric. Astronaut is designed to iteratively optimize the feature map structure to improve this primary metric.
