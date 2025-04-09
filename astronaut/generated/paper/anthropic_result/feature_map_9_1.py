import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class RyEnhancedSymmetricPhaseEncoder(BaseFeatureMap):
    """Ry-Enhanced Symmetric Phase Quantum Encoder (Refined).
    
    This feature map uses an optimized angle mapping with a refined linear scaling,
    an Ry-enhanced round-robin feature distribution, a standard three-layer entanglement,
    enhanced symmetric phase shifts, a strategic controlled-Z triplet pattern, a hybrid
    repetition structure, and an alternating Hadamard enhancement.
    
    Args:
        BaseFeatureMap (_type_): base feature map class
    
    Example:
        >>> feature_map = RyEnhancedSymmetricPhaseEncoder(n_qubits=10)
    """
    
    def __init__(self, n_qubits: int, scale_factor: float = np.pi, offset: float = np.pi/4) -> None:
        """Initialize the Ry-Enhanced Symmetric Phase Quantum Encoder.
        
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
        """Apply feature encoding with optimized angle mapping and Ry-enhanced distribution.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Define the distribution of features
        n_features = min(80, len(x))
        
        # Features 1-10 → Rx rotations on qubits 1-10
        for i in range(min(10, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 11-30 → Ry rotations on qubits 1-10 (two layers)
        for i in range(10, min(30, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 10) % self.n_qubits)
        
        # Features 31-40 → Rz rotations on qubits 1-10
        for i in range(30, min(40, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 30) % self.n_qubits)
        
        # Features 41-50 → Rx rotations on qubits 1-10 (second round)
        for i in range(40, min(50, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 40) % self.n_qubits)
        
        # Features 51-70 → Ry rotations on qubits 1-10 (two more layers)
        for i in range(50, min(70, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 50) % self.n_qubits)
        
        # Features 71-80 → Rz rotations on qubits 1-10 (second round)
        for i in range(70, min(80, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 70) % self.n_qubits)
    
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
        Rz(π/4) to even-indexed qubits and Rz(3π/4) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # Even-indexed qubits
                qml.RZ(phi=np.pi/4, wires=i)
            else:  # Odd-indexed qubits
                qml.RZ(phi=3*np.pi/4, wires=i)
    
    def _apply_phase2_symmetric(self) -> None:
        """Apply symmetric phase shifts after Layer 2:
        Rz(3π/4) to even-indexed qubits and Rz(π/4) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # Even-indexed qubits
                qml.RZ(phi=3*np.pi/4, wires=i)
            else:  # Odd-indexed qubits
                qml.RZ(phi=np.pi/4, wires=i)
    
    def _apply_phase3(self) -> None:
        """Apply phase shifts after Layer 3: Rz(π/2) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=np.pi/2, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.triplets:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])
    
    def _apply_alternating_hadamard(self) -> None:
        """Apply Hadamard gates to alternating qubits (0, 2, 4, 6, 8)."""
        for i in range(0, self.n_qubits, 2):
            qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Hybrid repetition structure with 2 repetitions
        for _ in range(2):
            # Encode features with Ry-enhanced distribution
            self._encode_features(x)
            
            # Apply three-layer entanglement with enhanced symmetric phase shifts
            self._apply_local_entanglement()
            self._apply_phase1_symmetric()
            
            self._apply_medium_entanglement()
            self._apply_phase2_symmetric()
            
            self._apply_global_entanglement()
            self._apply_phase3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Additional feature encoding layer
        self._encode_features(x)
        
        # Apply alternating Hadamard enhancement
        self._apply_alternating_hadamard()