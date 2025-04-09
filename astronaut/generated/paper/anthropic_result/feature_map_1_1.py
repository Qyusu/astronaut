import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class AdaptiveZZFeatureMap(BaseFeatureMap):
    """Adaptive ZZ-Feature Encoding with Alternating Entanglement.
    
    This feature map maps 80-dimensional PCA-reduced MNIST data to quantum states using linear scaling
    functions. Each feature xi is scaled using fixed parameters: θi = ai·xi + bi, where ai and bi are
    non-trainable scaling and offset factors.
    
    The mapping follows this procedure:
    1. Features are distributed across available qubits in a round-robin fashion
    2. After each rotation layer, an alternating entanglement pattern is applied:
       - For odd-numbered repetitions: Nearest-neighbor entanglement using CNOT gates in a ring topology
       - For even-numbered repetitions: Long-range entanglement with qubit i controlling qubit i+⌊n/2⌋ (mod n)
    3. Additionally, after every second repetition, three-qubit controlled-Z gates are applied to enhance expressivity
    4. The entire encoding-entanglement block is repeated D times
    
    Args:
        BaseFeatureMap (_type_): base feature map class
    
    Example:
        >>> feature_map = AdaptiveZZFeatureMap(n_qubits=10, repetitions=4)
    """

    def __init__(self, n_qubits: int, repetitions: int = 3, a: float = np.pi, b: float = 0.0) -> None:
        """Initialize the Adaptive ZZ-Feature Encoding feature map.

        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of repetitions of the encoding-entanglement block. Defaults to 3.
            a (float, optional): Scaling factor for feature values. Defaults to pi.
            b (float, optional): Offset factor for feature values. Defaults to 0.0.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        self.a: float = a
        self.b: float = b

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Calculate how many features each qubit needs to encode
        features_per_rep = 3 * self.n_qubits  # 3 rotations (Rx, Ry, Rz) per qubit
        
        for d in range(self.repetitions):
            # Apply rotations for feature encoding in round-robin fashion
            
            # First rotation layer (Rx)
            for i in range(self.n_qubits):
                feature_idx = d * features_per_rep + i
                if feature_idx < len(x):
                    angle = self.a * x[feature_idx] + self.b
                    qml.RX(phi=angle, wires=i)

            # Second rotation layer (Ry)
            for i in range(self.n_qubits):
                feature_idx = d * features_per_rep + self.n_qubits + i
                if feature_idx < len(x):
                    angle = self.a * x[feature_idx] + self.b
                    qml.RY(phi=angle, wires=i)

            # Third rotation layer (Rz)
            for i in range(self.n_qubits):
                feature_idx = d * features_per_rep + 2 * self.n_qubits + i
                if feature_idx < len(x):
                    angle = self.a * x[feature_idx] + self.b
                    qml.RZ(phi=angle, wires=i)

            # Apply alternating entanglement pattern
            if d % 2 == 0:  # Even-indexed repetition (odd-numbered repetition)
                # Nearest-neighbor entanglement in a ring topology
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Connect last qubit to first (ring topology)
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            else:  # Odd-indexed repetition (even-numbered repetition)
                # Long-range entanglement
                half_n = self.n_qubits // 2
                for i in range(self.n_qubits):
                    target = (i + half_n) % self.n_qubits
                    qml.CNOT(wires=[i, target])
            
            # Apply three-qubit controlled-Z gates after every second repetition
            if d % 2 == 1:  # Odd-indexed repetition (even-numbered repetition)
                for j in range(0, self.n_qubits - 2, 3):
                    # Apply a three-qubit controlled-Z using Toffoli gate with Hadamards
                    qml.Hadamard(wires=j+2)
                    qml.Toffoli(wires=[j, j+1, j+2])
                    qml.Hadamard(wires=j+2)