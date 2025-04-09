import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class SymmetricPhaseStructureEncoder(BaseFeatureMap):
    """Symmetric Phase Structure with Optimized Triplets feature map.
    
    This feature map creates a balanced, symmetric circuit design while 
    maintaining high-performing elements. It uses a refined linear scaling,
    round-robin feature distribution, three-layer entanglement, symmetric
    phase structure, optimized triplet pattern, hybrid repetition, and
    alternating Hadamard enhancement.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = SymmetricPhaseStructureEncoder(n_qubits=10)
    """

    def __init__(self, n_qubits: int, scale_factor: float = np.pi, offset: float = np.pi/4) -> None:
        """Initialize the Symmetric Phase Structure Encoder.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for rotation angles. Defaults to π.
            offset (float, optional): Offset value for rotation angles. Defaults to π/4.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        
        # Define global connection distance
        self.global_distance = max(1, n_qubits // 3)
        
        # Define triplets for controlled-Z gates
        self.triplets = []
        for i in range(self.n_qubits):
            self.triplets.append((i, (i + 3) % n_qubits, (i + 6) % n_qubits))

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with refined linear scaling.

        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Define rotation gates to use in sequence
        rotation_gates = [qml.RX, qml.RY, qml.RZ]
        
        # Distribute all features across qubits in a round-robin fashion
        for feature_idx in range(min(80, len(x))):
            qubit_idx = feature_idx % self.n_qubits
            rotation_type_idx = (feature_idx // self.n_qubits) % len(rotation_gates)
            rotation_gate = rotation_gates[rotation_type_idx]
            
            # Apply linear scaling: π·xi + π/4
            angle = self.scale_factor * x[feature_idx] + self.offset
            rotation_gate(phi=angle, wires=qubit_idx)

    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])

    def _apply_global_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by global distance (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.global_distance) % self.n_qubits])

    def _apply_phase1_symmetric(self) -> None:
        """Apply symmetric phase shifts after Layer 1: 
        Rz(π/4) to even-indexed qubits and Rz(π/3) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # Even-indexed qubits
                qml.RZ(phi=np.pi/4, wires=i)
            else:  # Odd-indexed qubits
                qml.RZ(phi=np.pi/3, wires=i)

    def _apply_phase2_symmetric(self) -> None:
        """Apply symmetric phase shifts after Layer 2: 
        Rz(π/3) to even-indexed qubits and Rz(π/4) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # Even-indexed qubits
                qml.RZ(phi=np.pi/3, wires=i)
            else:  # Odd-indexed qubits
                qml.RZ(phi=np.pi/4, wires=i)

    def _apply_phase3(self) -> None:
        """Apply phase shifts after Layer 3: Rz(π/2) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=np.pi/2, wires=i)

    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to optimized triplets."""
        for a, b, c in self.triplets:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])

    def _apply_alternating_hadamard(self) -> None:
        """Apply Hadamard gates to alternating qubits (0, 2, 4, 6, 8)."""
        for i in range(0, self.n_qubits, 2):  # Even indices
            qml.Hadamard(wires=i)

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Repeat encoding-entanglement block twice
        for _ in range(2):
            # Encode features
            self._encode_features(x)
            
            # Apply layers of entanglement with phase shifts
            self._apply_local_entanglement()
            self._apply_phase1_symmetric()
            
            self._apply_medium_entanglement()
            self._apply_phase2_symmetric()
            
            self._apply_global_entanglement()
            self._apply_phase3()
            
            # Apply optimized triplet pattern
            self._apply_cz_triplets()
        
        # Final feature encoding
        self._encode_features(x)
        
        # Apply alternating Hadamard enhancement
        self._apply_alternating_hadamard()