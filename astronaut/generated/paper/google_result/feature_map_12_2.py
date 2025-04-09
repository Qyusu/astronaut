import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LEFM_SFRXFeatureMap(BaseFeatureMap):
    """LEFM with Scaled Final RX layer (LEFM-SFRX) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, a_coeffs: np.ndarray = None, c_coeffs: np.ndarray = None, scaling_factor: float = 1.0) -> None:
        """Initialize the LEFM with Scaled Final RX layer feature map class.

        Args:
            n_qubits (int): number of qubits
            a_coeffs (np.ndarray): Coefficients for the first RX rotations. Default is np.ones(n_qubits).
            c_coeffs (np.ndarray): Coefficients for the final RX rotations. Default is np.ones(n_qubits).
            scaling_factor (float): Scaling factor for the final RX rotations. Default is 1.0.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.a_coeffs: np.ndarray = (a_coeffs if a_coeffs is not None else np.ones(n_qubits))
        self.c_coeffs: np.ndarray = (c_coeffs if c_coeffs is not None else np.ones(n_qubits))
        self.scaling_factor: float = scaling_factor

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        n_features = x.shape[0]
        n_layers = n_features // (3 * self.n_qubits)  # Each layer uses 3*n_qubits features

        for layer in range(n_layers):
            # First RX Encoding Layer
            for i in range(self.n_qubits):
                qml.RX(phi=self.a_coeffs[i] * x[layer * 3 * self.n_qubits + i], wires=i)

            # Entanglement Layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Quadratic RY Rotation Layer
            for i in range(self.n_qubits):
                qml.RY(phi=5 * x[layer * 3 * self.n_qubits + self.n_qubits + i] ** 2, wires=i)

            # Final RX Encoding Layer with Scaling
            for i in range(self.n_qubits):
                qml.RX(
                    phi=self.scaling_factor
                    * self.c_coeffs[i]
                    * x[layer * 3 * self.n_qubits + 2 * self.n_qubits + i],
                    wires=i,
                )
