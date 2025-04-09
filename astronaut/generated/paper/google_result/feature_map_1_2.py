import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class PauliZFeatureMap(BaseFeatureMap):
    """Pauli-Z Feature Map (PZFM) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int) -> None:
        """Initialize the Pauli-Z feature map class.

        Args:
            n_qubits (int): number of qubits
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Initialize all qubits to |0> state
        # No need for explicit initialization in PennyLane, qubits start in |0>

        # Apply ZZ gates for each pair of qubits
        feature_index = 0
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                # Ensure we don't run out of features
                if feature_index + 1 < len(x):
                  qml.IsingZZ(phi=x[feature_index] * x[feature_index + 1], wires=[i, j])
                  feature_index += 2
                else:
                  # If out of features, break the loop
                  break
            if feature_index +1 >= len(x):
              break