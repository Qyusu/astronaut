import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class DiverseTripletPatternEncoder(BaseFeatureMap):
    """Diverse Triplet Pattern Encoder feature map class.
    
    This feature map focuses on maximizing feature interaction diversity
    through strategically varied controlled-Z triplet patterns across circuit repetitions.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = DiverseTripletPatternEncoder(n_qubits=10)
    """

    def __init__(self, n_qubits: int, scale_factor: float = np.pi, offset: float = np.pi/4) -> None:
        """Initialize the Diverse Triplet Pattern Encoder feature map class.

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
        
        # Calculate global jump size for third layer entanglement
        self.global_jump = max(1, n_qubits // 3)
        
        # Define CZ triplet patterns for each repetition
        # First repetition: "skip-1" triplets
        self.skip1_triplets = []
        for i in range(n_qubits):
            self.skip1_triplets.append((i, (i + 2) % n_qubits, (i + 4) % n_qubits))
        
        # Second repetition: "skip-2" triplets
        self.skip2_triplets = []
        for i in range(n_qubits):
            self.skip2_triplets.append((i, (i + 3) % n_qubits, (i + 6) % n_qubits))

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with standard linear scaling.

        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        rotation_types = [qml.RX, qml.RY, qml.RZ]
        
        # Distribute all features across qubits in a round-robin fashion
        for feature_idx in range(min(80, len(x))):
            qubit_idx = feature_idx % self.n_qubits
            rotation_type_idx = (feature_idx // self.n_qubits) % 3
            rotation_gate = rotation_types[rotation_type_idx]
            
            # Apply standard linear scaling: π·xi + π/4
            angle = self.scale_factor * x[feature_idx] + self.offset
            rotation_gate(phi=angle, wires=qubit_idx)
    
    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits in a ring topology (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_phase_shift1(self) -> None:
        """Apply Rz(π/9) to odd-indexed qubits after Layer 1."""
        for i in range(1, self.n_qubits, 2):  # Odd indices
            qml.RZ(phi=np.pi/9, wires=i)
    
    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_phase_shift2(self) -> None:
        """Apply Rz(π/5) to even-indexed qubits after Layer 2."""
        for i in range(0, self.n_qubits, 2):  # Even indices
            qml.RZ(phi=np.pi/5, wires=i)
    
    def _apply_global_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance n/3 (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.global_jump) % self.n_qubits])
    
    def _apply_phase_shift3(self) -> None:
        """Apply Rz(3π/7) to all qubits after Layer 3."""
        for i in range(self.n_qubits):
            qml.RZ(phi=3*np.pi/7, wires=i)
    
    def _apply_cz_triplets(self, triplets) -> None:
        """Apply controlled-Z gates to specified triplets.
        
        Args:
            triplets (list): List of qubit triplets
        """
        for a, b, c in triplets:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])
    
    def _apply_final_hadamard(self) -> None:
        """Apply Hadamard gates to all even-indexed qubits."""
        for i in range(0, self.n_qubits, 2):  # Even indices
            qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # First repetition
        # Encode features
        self._encode_features(x)
        
        # Apply three-layer entanglement with phase shifts
        self._apply_local_entanglement()
        self._apply_phase_shift1()
        
        self._apply_medium_entanglement()
        self._apply_phase_shift2()
        
        self._apply_global_entanglement()
        self._apply_phase_shift3()
        
        # Apply first pattern of triplets
        self._apply_cz_triplets(self.skip1_triplets)
        
        # Second repetition
        # Encode features
        self._encode_features(x)
        
        # Apply three-layer entanglement with phase shifts
        self._apply_local_entanglement()
        self._apply_phase_shift1()
        
        self._apply_medium_entanglement()
        self._apply_phase_shift2()
        
        self._apply_global_entanglement()
        self._apply_phase_shift3()
        
        # Apply second pattern of triplets
        self._apply_cz_triplets(self.skip2_triplets)
        
        # Additional feature encoding layer at the end
        self._encode_features(x)
        
        # Apply final Hadamard layer to even-indexed qubits
        self._apply_final_hadamard()