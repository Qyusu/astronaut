import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class FineTunedAdaptiveAngleQuantumEncoder(BaseFeatureMap):
    """Fine-Tuned Adaptive Angle Quantum Encoder feature map class.
    
    This feature map optimizes rotation angle range, enhances entanglement patterns,
    and incorporates IQP-inspired elements for improved feature interaction.
    
    Args:
        BaseFeatureMap (_type_): base feature map class
    
    Example:
        >>> feature_map = FineTunedAdaptiveAngleQuantumEncoder(n_qubits=10)
    """

    def __init__(self, n_qubits: int, repetitions: int = 2) -> None:
        """Initialize the Fine-Tuned Adaptive Angle Quantum Encoder.
        
        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions. Defaults to 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        self.scale_factor: float = 0.95 * np.pi  # 0.95π
        self.offset: float = np.pi / 8  # π/8
        
        # Define optimized CZ triplets
        if n_qubits == 10:
            self.cz_triplets_set1 = [(0,2,5), (1,3,6), (2,4,7), (3,5,8), (4,6,9)]
            self.cz_triplets_set2 = [(5,8,1), (6,9,2), (7,0,3), (8,1,4), (9,2,5)]
        else:
            # For other qubit counts, generate a reasonable pattern
            self.cz_triplets_set1 = []
            self.cz_triplets_set2 = []
            for i in range(n_qubits):
                self.cz_triplets_set1.append((i, (i+2) % n_qubits, (i+5) % n_qubits))
                self.cz_triplets_set2.append(((i+5) % n_qubits, (i+8) % n_qubits, (i+1) % n_qubits))

    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply feature encoding with enhanced linear scaling.
        
        Distributes all 80 features across qubits in a round-robin fashion,
        applying Rx, Ry, and Rz rotations in sequence.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        rotation_types = [qml.RX, qml.RY, qml.RZ]
        
        # Iterate through all 80 features
        for feature_idx in range(80):
            # Determine which qubit to apply the rotation to
            qubit_idx = feature_idx % self.n_qubits
            
            # Determine which type of rotation to apply (Rx, Ry, or Rz)
            rotation_type_idx = (feature_idx // self.n_qubits) % 3
            rotation_gate = rotation_types[rotation_type_idx]
            
            # Calculate scaled angle: 0.95π·xi + π/8
            angle = self.scale_factor * x[feature_idx] + self.offset
            
            # Apply the selected rotation gate
            rotation_gate(phi=angle, wires=qubit_idx)
    
    def _apply_hadamard_layer(self) -> None:
        """Apply Hadamard gates to all qubits for IQP-inspired transformation.
        
        This creates quantum superposition that enhances the expressivity
        of the feature map.
        """
        for qubit_idx in range(self.n_qubits):
            qml.Hadamard(wires=qubit_idx)
    
    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits (local entanglement).
        
        Creates a ring topology where qubit i controls qubit i+1.
        """
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2.
        
        Creates connections between qubits that are not immediately adjacent.
        """
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_global_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 4.
        
        Creates long-range connections that complement local and medium-scale
        interactions.
        """
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 4) % self.n_qubits])
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to optimized triplets.
        
        Uses an optimized triplet pattern that ensures each qubit appears
        in multiple triplets, creating higher-order correlations between features.
        """
        # Apply CZ gates to first set of triplets
        for a, b, c in self.cz_triplets_set1:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[a, c])
        
        # Apply CZ gates to second set of triplets
        for a, b, c in self.cz_triplets_set2:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[a, c])

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Repeat the encoding-entanglement block the specified number of times
        for _ in range(self.repetitions):
            # Apply feature encoding
            self._apply_feature_encoding(x)
            
            # Apply IQP-inspired transformation
            self._apply_hadamard_layer()
            
            # Apply optimized three-layer entanglement
            self._apply_local_entanglement()
            self._apply_medium_entanglement()
            self._apply_global_entanglement()
            
            # Apply enhanced controlled-Z triplet selection
            self._apply_cz_triplets()