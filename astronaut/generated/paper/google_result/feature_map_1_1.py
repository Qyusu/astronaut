import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LinearEntangledFeatureMap(BaseFeatureMap):
    """Linear Entangled Feature Map (LEFM) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int) -> None:
        """Initialize the Linear Entangled feature map class.

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

        # Encode features using RX rotations and entangle with CNOT gates
        for k in range(80 // self.n_qubits):
            # Encode n features using RX gates
            for j in range(self.n_qubits):
                qml.RX(phi=x[k * self.n_qubits + j], wires=j)

            # Apply CNOT gates in a nearest-neighbor, circular fashion
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])  # Circular entanglement