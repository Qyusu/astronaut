import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class EnhancedLinearEntangledFeatureMap(BaseFeatureMap):
    """Enhanced Linear Entangled Feature Map (ELEFM) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, reps: int = 1) -> None:
        """Initialize the Enhanced Linear Entangled feature map class.

        Args:
            n_qubits (int): number of qubits
            reps (int): Number of repetitions (RX encoding followed by CNOT gates). Default is 1.
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

        for _ in range(self.reps):
            # RX encoding layer
            for i in range(self.n_qubits):
                qml.RX(phi=x[i % n_features], wires=i)

            # Full CNOT entanglement layer
            for i in range(self.n_qubits):
                for j in range(self.n_qubits):
                    if i != j:
                        qml.CNOT(wires=[i, j])

            # Additional RX layers based on available features
            for layer in range(self.n_qubits, n_features, self.n_qubits):
                for i in range(self.n_qubits):
                    qml.RX(phi=x[layer + i], wires=i)

                # Full CNOT entanglement layer
                for i in range(self.n_qubits):
                    for j in range(self.n_qubits):
                        if i != j:
                            qml.CNOT(wires=[i, j])