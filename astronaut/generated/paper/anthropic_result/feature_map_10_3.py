import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class MaximallyBalancedRyEncoderFeatureMap(BaseFeatureMap):
    """Maximally Balanced Ry Encoder with Optimized Angle Distribution feature map.

    This feature map creates a precisely tuned implementation with mathematically 
    optimized angle distributions, using 64% Ry gates and golden ratio enhanced entanglement.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = MaximallyBalancedRyEncoderFeatureMap(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.93 * np.pi, 
        offset: float = np.pi / 3.3,
        phase1_even: float = np.pi / 4,
        phase1_odd: float = np.pi / 2,
        phase2_even: float = np.pi / 8,
        phase2_odd: float = np.pi / 4,
        phase3: float = np.pi / 2,
        golden_ratio: float = 0.618,
        hadamard_phase: float = np.pi / 4,
        reps: int = 2
    ) -> None:
        """Initialize the Maximally Balanced Ry Encoder feature map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.93*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.3.
            phase1_even (float, optional): Phase shift for even qubits after layer 1. Defaults to np.pi/4.
            phase1_odd (float, optional): Phase shift for odd qubits after layer 1. Defaults to np.pi/2.
            phase2_even (float, optional): Phase shift for even qubits after layer 2. Defaults to np.pi/8.
            phase2_odd (float, optional): Phase shift for odd qubits after layer 2. Defaults to np.pi/4.
            phase3 (float, optional): Phase shift for all qubits after layer 3. Defaults to np.pi/2.
            golden_ratio (float, optional): Golden ratio for global entanglement. Defaults to 0.618.
            hadamard_phase (float, optional): Phase angle for final Hadamard layer. Defaults to np.pi/4.
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
        self.golden_ratio: float = golden_ratio
        self.hadamard_phase: float = hadamard_phase
        self.reps: int = reps
        
        # Calculate global entanglement distance based on golden ratio
        self.global_distance = max(1, int(n_qubits * golden_ratio))
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with precisely tuned Ry distribution (64% Ry gates).
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Ensure we don't go beyond the available features
        n_features = min(80, len(x))
        
        # Features 1-12 → Rx rotations on qubits 1-10 (with wrapping)
        for i in range(min(12, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 13-63 → Ry rotations on qubits 1-10 (5 complete layers + partial)
        for i in range(12, min(63, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 12) % self.n_qubits)
        
        # Features 64-80 → Rz rotations on qubits 1-10 (with partial)
        for i in range(63, min(80, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 63) % self.n_qubits)

    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_golden_global_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by golden ratio distance (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.global_distance) % self.n_qubits])
    
    def _apply_phase1(self) -> None:
        """Apply power-of-half phase angles after Layer 1:
        Rz(π/4) to even-indexed qubits and Rz(π/2) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_phase2(self) -> None:
        """Apply power-of-half phase angles after Layer 2:
        Rz(π/8) to even-indexed qubits and Rz(π/4) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_phase3(self) -> None:
        """Apply power-of-half phase angles after Layer 3:
        Rz(π/2) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_balanced_hadamard(self) -> None:
        """Apply perfectly balanced Hadamard pattern:
        - First half of qubits (0-4): Apply H gate then Rz(π/4)
        - Second half of qubits (5-9): Apply Rz(π/4) then H gate
        """
        half_qubits = self.n_qubits // 2
        
        # First half: H then Rz
        for i in range(half_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(phi=self.hadamard_phase, wires=i)
        
        # Second half: Rz then H
        for i in range(half_qubits, self.n_qubits):
            qml.RZ(phi=self.hadamard_phase, wires=i)
            qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Hybrid repetition structure
        for _ in range(self.reps):
            # Encode features with precisely tuned Ry distribution
            self._encode_features(x)
            
            # Apply entanglement layers with power-of-half phase angles
            self._apply_local_entanglement()
            self._apply_phase1()
            
            self._apply_medium_entanglement()
            self._apply_phase2()
            
            self._apply_golden_global_entanglement()
            self._apply_phase3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Additional feature encoding layer
        self._encode_features(x)
        
        # Apply perfectly balanced Hadamard pattern
        self._apply_balanced_hadamard()