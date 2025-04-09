import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LEFM_QPERY_b05FeatureMap(BaseFeatureMap):
    """LEFM with Quadratic Post-Entanglement RY (LEFM-QPERY-b0.5) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, a_coeffs: np.ndarray = None) -> None:
        """Initialize the LEFM with Quadratic Post-Entanglement RY feature map class.

        Args:
            n_qubits (int): number of qubits
            a_coeffs (np.ndarray): Coefficients for RX rotations. Default is np.ones(n_qubits).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.a_coeffs: np.ndarray = (a_coeffs if a_coeffs is not None else np.ones(n_qubits))

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        n_features = x.shape[0]
        n_layers = n_features // (2 * self.n_qubits)  # Each layer uses 2*n_qubits features

        for layer in range(n_layers):
            # RX Encoding Layer
            for i in range(self.n_qubits):
                qml.RX(phi=self.a_coeffs[i] * x[layer * 2 * self.n_qubits + i], wires=i)

            # Entanglement Layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Quadratic RY Rotation Layer
            for i in range(self.n_qubits):
                qml.RY(phi=0.5 * x[layer * 2 * self.n_qubits + self.n_qubits + i] ** 2, wires=i)
