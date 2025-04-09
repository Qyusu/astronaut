import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class OptimizedAngleRangeFeatureMap(BaseFeatureMap):
    """Optimized Angle Range with Enhanced Symmetry feature map.

    This feature map fine-tunes the angle range and implements a highly symmetric circuit design,
    including a modified linear scaling, Ry-enhanced feature distribution, three-layer entanglement,
    enhanced symmetric phase shifts, strategic CZ triplet pattern, and a symmetric Hadamard pattern.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = OptimizedAngleRangeFeatureMap(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.95 * np.pi, 
        offset: float = np.pi/3,
        phase1_even: float = np.pi/4,
        phase1_odd: float = np.pi/3,
        phase2_even: float = np.pi/3,
        phase2_odd: float = np.pi/4,
        phase3: float = np.pi/2,
        hadamard_phase1: float = np.pi/6,
        hadamard_phase2: float = np.pi/3,
        reps: int = 2
    ) -> None:
        """Initialize the Optimized Angle Range Feature Map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.95*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.
            phase1_even (float, optional): Phase shift for even qubits after layer 1. Defaults to np.pi/4.
            phase1_odd (float, optional): Phase shift for odd qubits after layer 1. Defaults to np.pi/3.
            phase2_even (float, optional): Phase shift for even qubits after layer 2. Defaults to np.pi/3.
            phase2_odd (float, optional): Phase shift for odd qubits after layer 2. Defaults to np.pi/4.
            phase3 (float, optional): Phase shift after layer 3. Defaults to np.pi/2.
            hadamard_phase1 (float, optional): Phase for first Hadamard group. Defaults to np.pi/6.
            hadamard_phase2 (float, optional): Phase for second Hadamard group. Defaults to np.pi/3.
            reps (int, optional): Number of repetitions. Defaults to 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        self.phase1_even: float = phase1_even
        self.phase1_odd: float = phase1_odd
        self.phase2_even: float = phase2_even
        self.phase2_odd: float = phase2_odd
        self.phase3: float = phase3
        self.hadamard_phase1: float = hadamard_phase1
        self.hadamard_phase2: float = hadamard_phase2
        self.reps: int = reps
        
        # Define global connection distance
        self.global_distance = max(1, n_qubits // 3)
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with Ry-enhanced distribution.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Ensure we don't go beyond the available features
        n_features = min(80, len(x))
        
        # Features 1-10 → Rx rotations on qubits 1-10
        for i in range(min(10, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 11-30 → Ry rotations on qubits 1-10 (first and second Ry layers)
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
        
        # Features 51-70 → Ry rotations on qubits 1-10 (third and fourth Ry layers)
        for i in range(50, min(70, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 50) % self.n_qubits)
        
        # Features 71-80 → Rz rotations on qubits 1-10 (second Rz layer)
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
        """Apply CNOT gates between qubits separated by distance n/3 (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.global_distance) % self.n_qubits])
    
    def _apply_phase1(self) -> None:
        """Apply phase shift after Layer 1: 
        Rz(π/4) to even-indexed qubits and Rz(π/3) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_phase2(self) -> None:
        """Apply phase shift after Layer 2: 
        Rz(π/3) to even-indexed qubits and Rz(π/4) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_phase3(self) -> None:
        """Apply phase shift after Layer 3: Rz(π/2) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_symmetric_hadamard(self) -> None:
        """Apply symmetric Hadamard pattern:
        - H gates to qubits 0, 3, 6, 9
        - Rz(π/6) followed by H gates to qubits 1, 4, 7
        - Rz(π/3) followed by H gates to qubits 2, 5, 8
        """
        # Apply H gates to qubits 0, 3, 6, 9 (group 1)
        for i in range(0, self.n_qubits, 3):
            qml.Hadamard(wires=i)
        
        # Apply Rz(π/6) followed by H gates to qubits 1, 4, 7 (group 2)
        for i in range(1, self.n_qubits, 3):
            qml.RZ(phi=self.hadamard_phase1, wires=i)
            qml.Hadamard(wires=i)
        
        # Apply Rz(π/3) followed by H gates to qubits 2, 5, 8 (group 3)
        for i in range(2, self.n_qubits, 3):
            qml.RZ(phi=self.hadamard_phase2, wires=i)
            qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Hybrid repetition structure
        for _ in range(self.reps):
            # Encode features with Ry-enhanced distribution
            self._encode_features(x)
            
            # Apply entanglement layers with enhanced symmetric phase shifts
            self._apply_local_entanglement()
            self._apply_phase1()
            
            self._apply_medium_entanglement()
            self._apply_phase2()
            
            self._apply_global_entanglement()
            self._apply_phase3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Additional feature encoding layer
        self._encode_features(x)
        
        # Apply symmetric Hadamard pattern
        self._apply_symmetric_hadamard()