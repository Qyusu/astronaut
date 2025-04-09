import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class ModifiedPauliZFeatureMap(BaseFeatureMap):
    """Modified Pauli-Z Feature Map (MPZFM) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, weights: float = 0.1) -> None:
        """Initialize the Modified Pauli-Z feature map class.

        Args:
            n_qubits (int): number of qubits
            weights (float): Pre-defined weights for ZZ gate angles. Default is 0.1.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.weights: float = weights
        self.feature_map_count = 0

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        n_features = x.shape[0]

        # Apply ZZ gates between all pairs of qubits
        k = 0  # Feature index
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                # Determine the angle theta_{i,j} using a single feature x_k
                theta = self.weights * x[k % n_features]  # Use modulo to cycle through features
                qml.IsingZZ(phi=theta, wires=[i, j])
                k += 1
        self.feature_map_count += 1