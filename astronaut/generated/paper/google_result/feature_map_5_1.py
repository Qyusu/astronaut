import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LEFMSPERYFeatureMap(BaseFeatureMap):
    """LEFM with Scaled Post-Entanglement RY (LEFM-SPERY) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, scaling_factor: float = 0.5, a_coeffs: np.ndarray = None, b_coeffs: np.ndarray = None) -> None:
        """Initialize the LEFM with Scaled Post-Entanglement RY feature map class.

        Args:
            n_qubits (int): number of qubits
            scaling_factor (float): Scaling factor for RY rotations. Default is 0.5.
            a_coeffs (np.ndarray): Coefficients for RX rotations. Default is np.ones(n_qubits).
            b_coeffs (np.ndarray): Coefficients for RY rotations. Default is np.ones(n_qubits).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scaling_factor: float = scaling_factor
        self.a_coeffs: np.ndarray = (a_coeffs if a_coeffs is not None else np.ones(n_qubits))
        self.b_coeffs: np.ndarray = (b_coeffs if b_coeffs is not None else np.ones(n_qubits))

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        n_features = x.shape[0]
        n_layers = n_features // (2*self.n_qubits) # Each layer uses 2*n_qubits features

        for layer in range(n_layers):
            # RX Encoding Layer
            for i in range(self.n_qubits):
                qml.RX(phi=self.a_coeffs[i] * x[layer * 2 * self.n_qubits + i], wires=i)

            # Entanglement Layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Scaled RY Rotation Layer
            for i in range(self.n_qubits):
                qml.RY(phi=self.scaling_factor * self.b_coeffs[i] * x[layer * 2 * self.n_qubits + self.n_qubits + i], wires=i)