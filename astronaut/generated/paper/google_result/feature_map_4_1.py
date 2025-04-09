import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LEFMPERFeatureMap(BaseFeatureMap):
    """LEFM with Post-Entanglement Rotations (LEFM-PER) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, a_coeff: float = 1.0, b_coeff: float = 1.0, c_coeff: float = 1.0) -> None:
        """Initialize the LEFM with Post-Entanglement Rotations feature map class.

        Args:
            n_qubits (int): number of qubits
            a_coeff (float): Coefficient for RX rotations. Default is 1.0.
            b_coeff (float): Coefficient for RY rotations. Default is 1.0.
            c_coeff (float): Coefficient for RZ rotations. Default is 1.0.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.a_coeff: float = a_coeff
        self.b_coeff: float = b_coeff
        self.c_coeff: float = c_coeff

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        n_features = x.shape[0]
        n_layers = n_features // (3 * self.n_qubits)

        for layer in range(n_layers):
            # RX Encoding Layer
            for i in range(self.n_qubits):
                qml.RX(phi=self.a_coeff * x[layer * 3 * self.n_qubits + i], wires=i)

            # Entanglement Layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])

            # RY Rotation Layer
            for i in range(self.n_qubits):
                qml.RY(phi=self.b_coeff * x[layer * 3 * self.n_qubits + self.n_qubits + i], wires=i)

            # RZ Rotation Layer
            for i in range(self.n_qubits):
                qml.RZ(phi=self.c_coeff * x[layer * 3 * self.n_qubits + 2 * self.n_qubits + i], wires=i)