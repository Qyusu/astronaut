from textwrap import dedent

REFLECTION_EXPERTISE = "quantum computing and machine learning"

FEATURE_MAP_CONSTRAINTS = dedent(
    """The quantum feature maps (\\Phi(x)) will be used to compute the quantum kernel K(x, x') = |\\langle\\Phi(x)|\\Phi(x')\\rangle|^2 for a QSVM (Quantum Support Vector Machine). Ensure the designs are tailored to this kernel computation.
    1. **Design Considerations**:
        - Define combinations of quantum gates.
        - Define an entanglement pattern for the features.
        - Specify the method for embedding input data and quantum states as rotation angles of quantum gates.
        - Ensure the high-dimensional input data is utilized effectively, minimizing any loss of information.
            - Avoid excessive feature compression that may lead to information loss (e.g., simple feature averaging, summing, etc.).
            - Ensure that information from **all dimensions** of the input data can be utilized effectively.
    2. **Restrictions on Encoding and Embedding**:
        - Only **linear functions** are allowed for encoding and embedding.
        - All parameters in the encoding and embedding must be **non-trainable**.
    3. **Execution Environment**:
        - Feature map evaluations are performed on ideal quantum simulators without noise, so hardware noise and qubit placement constraints do not need to be considered.
        - Propose a quantum feature map that is independent of the number of qubits in the quantum device.

    ## Hyperparameter Design Guidelines
    When designing feature map architectures, follow these principles for hyperparameter management:

    Note: Hyperparameters refer to the parameters defined in the `__init__` method of the feature map Python class, which determine the structure and behavior of the quantum circuit before training begins.

    - **Automatic calculation**: Feature map hyperparameters should be automatically computed from the input data or domain knowledge whenever possible to minimize user configuration burden
    - **User-configurable parameters**: If user-settable hyperparameters are necessary, explicitly specify:
        - **Purpose**: Clear explanation of what the parameter controls and its impact on performance
        - **Setting method**: Concrete instructions on how to determine appropriate values
        - **Default value**: Recommended default with justification based on typical input data or domain knowledge
    """
)

PERFORMANCE_METRIC_DEFINITION = dedent(
    """The performance metric is defined as the mean accuracy across multiple datasets.

    The performance metric is given by the following formula:
        Performance = \\sum_{{i}} Accuracy_i / D

    Where:
    - D is the number of datasets.
    - Accuracy_i is the accuracy of the i-th dataset (range: 0.0 to 1.0).
    """
)

INPUT_DATA_CONSTRAINTS = dedent(
    """The input data, originally represented as 784-dimensional image data, has been compressed to 80 dimensions using PCA and each value normalized to the range [0.0, 1.0].
    """
)
