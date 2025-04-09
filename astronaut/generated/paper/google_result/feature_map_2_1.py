import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class CombinedRotationFeatureMap(BaseFeatureMap):
    """Combined Rotation Feature Map (CRFM) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, reps: int = 1) -> None:
        """Initialize the Combined Rotation feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): Number of repetitions of the sequence. Default is 1.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.reps: int = reps

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        n_features = len(x)
        n_layers = 3  # RX, RY, RZ layers
        features_per_rep = self.n_qubits * n_layers
        total_features_needed = features_per_rep * self.reps

        if total_features_needed > n_features:
            raise ValueError(
                f"Not enough features provided. The feature map requires {total_features_needed} features, but only {n_features} were provided."
            )

        for rep in range(self.reps):
            feature_index = rep * features_per_rep

            # RX, RY, and RZ layers
            for i in range(self.n_qubits):
                qml.RX(phi=x[feature_index % n_features], wires=i)
                feature_index += 1
                qml.RY(phi=x[feature_index % n_features], wires=i)
                feature_index += 1
                qml.RZ(phi=x[feature_index % n_features], wires=i)
                feature_index += 1

            # Entanglement layer
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])