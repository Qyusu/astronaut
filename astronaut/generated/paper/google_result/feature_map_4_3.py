import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LEFMMEFeatureMap(BaseFeatureMap):
    """LEFM with Modified Entanglement (LEFM-ME) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, offset: int = None) -> None:
        """Initialize the LEFM with Modified Entanglement feature map class.

        Args:
            n_qubits (int): number of qubits
            offset (int): Offset for CNOT gates. Default is n_qubits // 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.offset: int = offset if offset is not None else n_qubits // 2

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        n_features = x.shape[0]
        n_layers = n_features // self.n_qubits

        for layer in range(n_layers):
            # RX Encoding Layer
            for i in range(self.n_qubits):
                qml.RX(phi=x[layer * self.n_qubits + i], wires=i)

            # Entanglement Layer with Offset
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + self.offset) % self.n_qubits])