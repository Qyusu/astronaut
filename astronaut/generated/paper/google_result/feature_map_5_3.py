import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LEFM_SPERY2FeatureMap(BaseFeatureMap):
    """LEFM with Selective Post-Entanglement RY (LEFM-SPERY2) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, a_coeffs: np.ndarray = None, b_coeffs: np.ndarray = None, selected_qubits: list[int] = None) -> None:
        """Initialize the LEFM with Selective Post-Entanglement RY feature map class.

        Args:
            n_qubits (int): number of qubits
            a_coeffs (np.ndarray): Coefficients for RX rotations. Default is np.ones(n_qubits).
            b_coeffs (np.ndarray): Coefficients for RY rotations. Default is np.ones(len(selected_qubits)).
            selected_qubits (list[int]): Indices of qubits to apply RY rotations. Default is range(0, n_qubits, 2) (even-numbered qubits).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.a_coeffs: np.ndarray = (a_coeffs if a_coeffs is not None else np.ones(n_qubits))
        self.selected_qubits: list[int] = (selected_qubits if selected_qubits is not None else list(range(0, n_qubits, 2)))
        self.b_coeffs: np.ndarray = (b_coeffs if b_coeffs is not None else np.ones(len(self.selected_qubits)))

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

            # Selective RY Rotation Layer
            selected_index = 0
            for i in self.selected_qubits:
                qml.RY(phi=self.b_coeffs[selected_index] * x[layer * 2 * self.n_qubits + self.n_qubits + selected_index], wires=i)
                selected_index +=1
