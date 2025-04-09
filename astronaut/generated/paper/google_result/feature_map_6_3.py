import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LEFM_MNLPERYFeatureMap(BaseFeatureMap):
    """LEFM with Mixed Non-linear Post-Entanglement RY (LEFM-MNLPERY) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, a_coeffs: np.ndarray = None, b_coeff: float = 1.0, c_coeff: float = 1.0, subset1: list = None, subset2: list = None) -> None:
        """Initialize the LEFM with Mixed Non-linear Post-Entanglement RY feature map class.

        Args:
            n_qubits (int): number of qubits
            a_coeffs (np.ndarray): Coefficients for RX rotations. Default is np.ones(n_qubits).
            b_coeff (float): Coefficient for the sine function in RY rotations. Default is 1.0.
            c_coeff (float): Coefficient for the quadratic function in RY rotations. Default is 1.0.
            subset1 (list): List of qubit indices for sine function. Default is [0, 2, 4, 6, 8].
            subset2 (list): List of qubit indices for quadratic function. Default is [1, 3, 5, 7, 9].
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.a_coeffs: np.ndarray = (a_coeffs if a_coeffs is not None else np.ones(n_qubits))
        self.b_coeff: float = b_coeff
        self.c_coeff: float = c_coeff
        self.subset1: list = (subset1 if subset1 is not None else list(range(0, n_qubits, 2)))
        self.subset2: list = (subset2 if subset2 is not None else list(range(1, n_qubits, 2)))

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

            # Mixed Non-linear RY Rotation Layer
            for i in range(self.n_qubits):
                if i in self.subset1:
                    qml.RY(phi=np.sin(self.b_coeff * x[layer * 2 * self.n_qubits + self.n_qubits + i]), wires=i)
                elif i in self.subset2:
                    qml.RY(phi=self.c_coeff * x[layer * 2 * self.n_qubits + self.n_qubits + i] ** 2, wires=i)
                else:
                    # This should never happen, but it's good practice to have a default
                    pass
