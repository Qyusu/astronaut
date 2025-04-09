import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class PhaseEnhancedThreeLayerEncoder(BaseFeatureMap):
    """Phase-Enhanced Three-Layer Encoder feature map class.
    
    This feature map builds upon the effective three-layer structure while introducing 
    strategic phase shifts and a hybrid repetition approach.
    
    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = PhaseEnhancedThreeLayerEncoder(n_qubits=10)
    """

    def __init__(self, n_qubits: int, repetitions: int = 2) -> None:
        """Initialize the Phase-Enhanced Three-Layer Encoder feature map class.

        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions. Defaults to 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        self.scale_factor: float = np.pi  # π
        self.offset: float = np.pi / 4  # π/4
        
        # Phase shift angles
        self.phase_shift_1: float = np.pi / 8  # π/8
        self.phase_shift_2: float = np.pi / 4  # π/4
        self.phase_shift_3: float = 3 * np.pi / 8  # 3π/8
        
        # Global entanglement distance (n/3 rounded)
        self.global_distance: int = max(1, n_qubits // 3)
        
        # Define CZ triplet patterns
        if n_qubits == 10:
            self.cz_triplets_set1 = [(0,3,6), (1,4,7), (2,5,8), (3,6,9), (4,7,0)]
            self.cz_triplets_set2 = [(5,8,1), (6,9,2), (7,0,3), (8,1,4), (9,2,5)]
        else:
            # For other qubit counts, generate a reasonable pattern
            self.cz_triplets_set1 = []
            self.cz_triplets_set2 = []
            for i in range(n_qubits):
                self.cz_triplets_set1.append((i, (i+3) % n_qubits, (i+6) % n_qubits))
                self.cz_triplets_set2.append(((i+5) % n_qubits, (i+8) % n_qubits, (i+1) % n_qubits))

    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply feature encoding with standard linear scaling.
        
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
            
            # Calculate scaled angle: π·xi + π/4
            angle = self.scale_factor * x[feature_idx] + self.offset
            
            # Apply the selected rotation gate
            rotation_gate(phi=angle, wires=qubit_idx)
    
    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits (local entanglement).
        
        Creates a ring topology where qubit i controls qubit i+1.
        """
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_phase_shift_1(self) -> None:
        """Apply Rz(π/8) phase shifts to odd-indexed qubits after local entanglement."""
        for i in range(self.n_qubits):
            if i % 2 == 1:  # odd-indexed qubits
                qml.RZ(phi=self.phase_shift_1, wires=i)
    
    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2.
        
        Creates connections between qubits that are not immediately adjacent.
        """
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_phase_shift_2(self) -> None:
        """Apply Rz(π/4) phase shifts to even-indexed qubits after medium entanglement."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even-indexed qubits
                qml.RZ(phi=self.phase_shift_2, wires=i)
    
    def _apply_global_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance n/3.
        
        Creates long-range connections for diverse feature interactions.
        """
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.global_distance) % self.n_qubits])
    
    def _apply_phase_shift_3(self) -> None:
        """Apply Rz(3π/8) phase shifts to all qubits after global entanglement."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase_shift_3, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets.
        
        Uses a pattern that ensures each qubit appears in multiple triplets,
        creating higher-order correlations between features.
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
            
            # Apply modified three-layer entanglement with strategic phase shifts
            self._apply_local_entanglement()
            self._apply_phase_shift_1()
            
            self._apply_medium_entanglement()
            self._apply_phase_shift_2()
            
            self._apply_global_entanglement()
            self._apply_phase_shift_3()
            
            # Apply strategic controlled-Z gates
            self._apply_cz_triplets()
        
        # Add final feature encoding layer (without entanglement)
        self._apply_feature_encoding(x)