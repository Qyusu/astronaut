import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class GoldenRatioEncoder(BaseFeatureMap):
    """Golden Ratio Quantum Encoder with Enhanced Ry feature map.

    This feature map increases the proportion of Ry gates and implements
    phase relationships based on the golden ratio.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = GoldenRatioEncoder(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.94 * np.pi, 
        offset: float = np.pi / 3.35,
        golden_ratio: float = 1.618,
        phase1_even: float = None,  # Will be set to π/φ in __init__
        phase1_odd: float = None,   # Will be set to π/φ² in __init__
        phase2_even: float = None,  # Will be set to π/φ² in __init__
        phase2_odd: float = None,   # Will be set to π/φ in __init__
        phase3_all: float = np.pi / 2,
        h_mod_phase1: float = np.pi / 6,
        h_mod_phase2: float = np.pi / 2,
        h_mod_phase3: float = 5 * np.pi / 6,
        reps: int = 2
    ) -> None:
        """Initialize the Golden Ratio Quantum Encoder with Enhanced Ry feature map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.94*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.35.
            golden_ratio (float, optional): The golden ratio value. Defaults to 1.618.
            phase1_even (float, optional): Phase for even qubits after layer 1. Defaults to π/φ.
            phase1_odd (float, optional): Phase for odd qubits after layer 1. Defaults to π/φ².
            phase2_even (float, optional): Phase for even qubits after layer 2. Defaults to π/φ².
            phase2_odd (float, optional): Phase for odd qubits after layer 2. Defaults to π/φ.
            phase3_all (float, optional): Phase for all qubits after layer 3. Defaults to π/2.
            h_mod_phase1 (float, optional): Phase for mod 4 = 1 qubits before Hadamard. Defaults to π/6.
            h_mod_phase2 (float, optional): Phase for mod 4 = 2 qubits before Hadamard. Defaults to π/2.
            h_mod_phase3 (float, optional): Phase for mod 4 = 3 qubits before Hadamard. Defaults to 5π/6.
            reps (int, optional): Number of repetitions. Defaults to 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        self.golden_ratio: float = golden_ratio
        
        # Calculate golden ratio phases if not provided
        self.phase1_even = phase1_even if phase1_even is not None else np.pi / golden_ratio
        self.phase1_odd = phase1_odd if phase1_odd is not None else np.pi / (golden_ratio**2)
        self.phase2_even = phase2_even if phase2_even is not None else np.pi / (golden_ratio**2)
        self.phase2_odd = phase2_odd if phase2_odd is not None else np.pi / golden_ratio
        
        self.phase3_all: float = phase3_all
        self.h_mod_phase1: float = h_mod_phase1
        self.h_mod_phase2: float = h_mod_phase2
        self.h_mod_phase3: float = h_mod_phase3
        self.reps: int = reps
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features_first_rep(self, x: np.ndarray) -> None:
        """Apply feature encoding for the first repetition.
        
        First repetition (30 features):
        * Features 1-6 → Rx rotations on qubits 1-6
        * Features 7-26 → Ry rotations on qubits 7-10 and 1-16 (exactly 20 Ry gates)
        * Features 27-30 → Rz rotations on qubits 7-10
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 1-6 → Rx rotations on qubits 1-6 (0-5 in 0-indexed)
        for i in range(min(6, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 7-26 → Ry rotations on qubits 7-10 and 1-16 (exactly 20 Ry gates)
        for i in range(6, min(26, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map to qubits 7-10 (6-9 in 0-indexed) and then 1-16 (0-15 in 0-indexed)
            if i < 10:  # For features 7-10 map to qubits 7-10 (6-9 in 0-indexed)
                wire_idx = i
            else:  # For features 11-26 map to qubits 1-16 (0-15 in 0-indexed)
                wire_idx = (i - 10) % self.n_qubits
            qml.RY(phi=angle, wires=wire_idx % self.n_qubits)
        
        # Features 27-30 → Rz rotations on qubits 7-10 (6-9 in 0-indexed)
        for i in range(26, min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 26 + 6) % self.n_qubits  # Map to qubits 7-10 (6-9 in 0-indexed)
            qml.RZ(phi=angle, wires=wire_idx)

    def _encode_features_second_rep(self, x: np.ndarray) -> None:
        """Apply feature encoding for the second repetition.
        
        Second repetition (30 features):
        * Features 31-36 → Rx rotations on qubits 5-10
        * Features 37-56 → Ry rotations on qubits 1-20 (exactly 20 Ry gates)
        * Features 57-60 → Rz rotations on qubits 1-4
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 31-36 → Rx rotations on qubits 5-10 (4-9 in 0-indexed)
        for i in range(30, min(36, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 30 + 4) % self.n_qubits  # Map to qubits 5-10 (4-9 in 0-indexed)
            qml.RX(phi=angle, wires=wire_idx)
        
        # Features 37-56 → Ry rotations on qubits 1-20 (0-19 in 0-indexed)
        for i in range(36, min(56, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 36) % self.n_qubits  # Map to qubits 1-20 (0-19 in 0-indexed)
            qml.RY(phi=angle, wires=wire_idx)
        
        # Features 57-60 → Rz rotations on qubits 1-4 (0-3 in 0-indexed)
        for i in range(56, min(60, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 56) % self.n_qubits  # Map to qubits 1-4 (0-3 in 0-indexed)
            qml.RZ(phi=angle, wires=wire_idx)

    def _encode_final_layer(self, x: np.ndarray) -> None:
        """Apply feature encoding for the final layer.
        
        Final encoding layer (20 features):
        * Features 61-64 → Rx rotations on qubits 7-10
        * Features 65-78 → Ry rotations on qubits 1-14 (exactly 14 Ry gates)
        * Features 79-80 → Rz rotations on qubits 5-6
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 61-64 → Rx rotations on qubits 7-10 (6-9 in 0-indexed)
        for i in range(60, min(64, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 60 + 6) % self.n_qubits  # Map to qubits 7-10 (6-9 in 0-indexed)
            qml.RX(phi=angle, wires=wire_idx)
        
        # Features 65-78 → Ry rotations on qubits 1-14 (0-13 in 0-indexed)
        for i in range(64, min(78, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 64) % self.n_qubits  # Map to qubits 1-14 (0-13 in 0-indexed)
            qml.RY(phi=angle, wires=wire_idx)
        
        # Features 79-80 → Rz rotations on qubits 5-6 (4-5 in 0-indexed)
        for i in range(78, min(80, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 78 + 4) % self.n_qubits  # Map to qubits 5-6 (4-5 in 0-indexed)
            qml.RZ(phi=angle, wires=wire_idx)

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
        distance = max(1, self.n_qubits // 3)  # Ensure distance is at least 1
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + distance) % self.n_qubits])
    
    def _apply_golden_phase1(self) -> None:
        """Apply Golden Ratio Phase pattern after Layer 1:
        Rz(π/φ) to even-indexed qubits and Rz(π/φ²) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_golden_phase2(self) -> None:
        """Apply Golden Ratio Phase pattern after Layer 2:
        Rz(π/φ²) to even-indexed qubits and Rz(π/φ) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_phase3(self) -> None:
        """Apply Phase pattern after Layer 3:
        Rz(π/2) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3_all, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_fourier_hadamard(self) -> None:
        """Apply Fourier-Inspired Hadamard Pattern:
        - Qubit index mod 4 = 0: Apply H gate
        - Qubit index mod 4 = 1: Apply Rz(π/6) followed by H gate
        - Qubit index mod 4 = 2: Apply Rz(π/2) followed by H gate
        - Qubit index mod 4 = 3: Apply Rz(5π/6) followed by H gate
        """
        for i in range(self.n_qubits):
            mod4 = i % 4
            if mod4 == 0:
                qml.Hadamard(wires=i)
            elif mod4 == 1:
                qml.RZ(phi=self.h_mod_phase1, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 2:
                qml.RZ(phi=self.h_mod_phase2, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 3:
                qml.RZ(phi=self.h_mod_phase3, wires=i)
                qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # First repetition
        self._encode_features_first_rep(x)
        
        # Apply entanglement layers with Golden Phase pattern
        self._apply_local_entanglement()
        self._apply_golden_phase1()
        
        self._apply_medium_entanglement()
        self._apply_golden_phase2()
        
        self._apply_global_entanglement()
        self._apply_phase3()
        
        # Apply controlled-Z triplet pattern
        self._apply_cz_triplets()
        
        # Second repetition
        self._encode_features_second_rep(x)
        
        # Apply entanglement layers with Golden Phase pattern
        self._apply_local_entanglement()
        self._apply_golden_phase1()
        
        self._apply_medium_entanglement()
        self._apply_golden_phase2()
        
        self._apply_global_entanglement()
        self._apply_phase3()
        
        # Apply controlled-Z triplet pattern
        self._apply_cz_triplets()
        
        # Apply final encoding layer
        self._encode_final_layer(x)
        
        # Apply Fourier-Inspired Hadamard Pattern
        self._apply_fourier_hadamard()