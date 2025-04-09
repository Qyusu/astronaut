import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class StrategicCZEnhancedQuantumEncoder(BaseFeatureMap):
    """Strategic CZ-Enhanced Quantum Encoder feature map class.

    This feature map emphasizes controlled-Z gates in creating higher-order correlations
    between features with a hierarchical structure.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = StrategicCZEnhancedQuantumEncoder(n_qubits=10)
    """

    def __init__(self, n_qubits: int, repetitions: int = 3) -> None:
        """Initialize the Strategic CZ-Enhanced Quantum Encoder.

        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions. Defaults to 3.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        self.scale_factor: float = 3 * np.pi / 2  # 3π/2
        self.offset: float = np.pi / 4  # π/4

    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply feature encoding with bounded linear scaling.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Distribute 80 features in round-robin fashion
        for feature_idx in range(80):
            # Determine which qubit to apply the rotation to
            qubit_idx = feature_idx % self.n_qubits
            
            # Calculate scaled angle: 3π/2·xi + π/4
            angle = self.scale_factor * x[feature_idx] + self.offset
            
            # Alternate between Rx and Rz rotations
            if (feature_idx // self.n_qubits) % 2 == 0:
                # Even rounds use Rx
                qml.RX(phi=angle, wires=qubit_idx)
            else:
                # Odd rounds use Rz
                qml.RZ(phi=angle, wires=qubit_idx)
    
    def _apply_cnot_ring(self) -> None:
        """Apply CNOT gates in a ring topology."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_cz_layer1(self) -> None:
        """Apply CZ gates for local correlations (adjacent triplets)."""
        # For a 10-qubit system, define specific triplets
        if self.n_qubits == 10:
            triplets = [(0,1,2), (2,3,4), (4,5,6), (6,7,8), (8,9,0)]
        else:
            # For other qubit counts, generate appropriate triplets
            triplets = []
            for i in range(0, self.n_qubits, 2):
                triplets.append((i, (i + 1) % self.n_qubits, (i + 2) % self.n_qubits))
        
        # Apply CZ gates to each triplet
        for a, b, c in triplets:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[a, c])
    
    def _apply_cz_layer2(self) -> None:
        """Apply CZ gates for medium-range correlations (skip-1 triplets)."""
        # For a 10-qubit system, define specific triplets
        if self.n_qubits == 10:
            triplets = [(0,2,4), (1,3,5), (2,4,6), (3,5,7), (4,6,8), 
                       (5,7,9), (6,8,0), (7,9,1), (8,0,2), (9,1,3)]
        else:
            # For other qubit counts, generate appropriate triplets
            triplets = []
            for i in range(self.n_qubits):
                triplets.append((i, (i + 2) % self.n_qubits, (i + 4) % self.n_qubits))
        
        # Apply CZ gates to each triplet
        for a, b, c in triplets:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[a, c])
    
    def _apply_cz_layer3(self) -> None:
        """Apply CZ gates for long-range correlations (skip-2 triplets)."""
        # For a 10-qubit system, define specific triplets
        if self.n_qubits == 10:
            triplets = [(0,3,6), (1,4,7), (2,5,8), (3,6,9), (4,7,0), 
                       (5,8,1), (6,9,2), (7,0,3), (8,1,4), (9,2,5)]
        else:
            # For other qubit counts, generate appropriate triplets
            triplets = []
            for i in range(self.n_qubits):
                triplets.append((i, (i + 3) % self.n_qubits, (i + 6) % self.n_qubits))
        
        # Apply CZ gates to each triplet
        for a, b, c in triplets:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[a, c])

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Initialize qubits with Hadamard gates for superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        
        # Repeat the encoding-entanglement block the specified number of times
        for _ in range(self.repetitions):
            # Apply feature encoding
            self._apply_feature_encoding(x)
            
            # Apply foundational CNOT ring entanglement
            self._apply_cnot_ring()
            
            # Apply hierarchical CZ gate network
            self._apply_cz_layer1()  # Local correlations
            self._apply_cz_layer2()  # Medium-range correlations
            self._apply_cz_layer3()  # Long-range correlations