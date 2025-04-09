import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class PerfectlyBalancedRyWithComplementaryPhases(BaseFeatureMap):
    """Perfectly Balanced Ry Distribution with Complementary Phases feature map.

    This feature map creates an exact balance of Ry gates and implements complementary 
    phase patterns for even and odd qubits.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = PerfectlyBalancedRyWithComplementaryPhases(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.93 * np.pi, 
        offset: float = np.pi / 3.4,
        phase1_even: float = np.pi / 3,
        phase1_odd: float = np.pi / 2,
        phase2_even: float = np.pi / 2,
        phase2_odd: float = np.pi / 3,
        phase3_all: float = np.pi / 4,
        h_mod_phase1: float = np.pi / 5,
        h_mod_phase2: float = np.pi / 2,
        h_mod_phase3: float = 4 * np.pi / 5,
        reps: int = 2
    ) -> None:
        """Initialize the Perfectly Balanced Ry Distribution with Complementary Phases feature map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.93*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.4.
            phase1_even (float, optional): Phase for even qubits after layer 1. Defaults to np.pi/3.
            phase1_odd (float, optional): Phase for odd qubits after layer 1. Defaults to np.pi/2.
            phase2_even (float, optional): Phase for even qubits after layer 2. Defaults to np.pi/2.
            phase2_odd (float, optional): Phase for odd qubits after layer 2. Defaults to np.pi/3.
            phase3_all (float, optional): Phase for all qubits after layer 3. Defaults to np.pi/4.
            h_mod_phase1 (float, optional): Phase for mod 4 = 1 qubits before Hadamard. Defaults to np.pi/5.
            h_mod_phase2 (float, optional): Phase for mod 4 = 2 qubits before Hadamard. Defaults to np.pi/2.
            h_mod_phase3 (float, optional): Phase for mod 4 = 3 qubits before Hadamard. Defaults to 4*np.pi/5.
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
        * Features 7-24 → Ry rotations on qubits 7-10 and 1-14 (exactly 18 Ry gates)
        * Features 25-30 → Rz rotations on qubits 5-10
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 1-6 → Rx rotations on qubits 1-6
        for i in range(min(6, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i + 1) % self.n_qubits)
        
        # Features 7-24 → Ry rotations on qubits 7-10 and 1-14 (exactly 18 Ry gates)
        for i in range(6, min(24, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # For i = 6 to 9 (features 7-10), map to qubits 7-10
            # For i = 10 to 23 (features 11-24), map to qubits 1-14
            if i < 10:
                wire_idx = (i - 6 + 7) % self.n_qubits  # Maps 6-9 to 7-10
            else:
                wire_idx = ((i - 10) % 14 + 1) % self.n_qubits  # Maps 10-23 to 1-14
            qml.RY(phi=angle, wires=wire_idx)
        
        # Features 25-30 → Rz rotations on qubits 5-10
        for i in range(24, min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = ((i - 24) + 5) % self.n_qubits  # Maps 24-29 to 5-10
            qml.RZ(phi=angle, wires=wire_idx)

    def _encode_features_second_rep(self, x: np.ndarray) -> None:
        """Apply feature encoding for the second repetition.
        
        Second repetition (30 features):
        * Features 31-36 → Rx rotations on qubits 5-10
        * Features 37-54 → Ry rotations on qubits 1-18 (exactly 18 Ry gates)
        * Features 55-60 → Rz rotations on qubits 1-6
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 31-36 → Rx rotations on qubits 5-10
        for i in range(30, min(36, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = ((i - 30) + 5) % self.n_qubits  # Maps 30-35 to 5-10
            qml.RX(phi=angle, wires=wire_idx)
        
        # Features 37-54 → Ry rotations on qubits 1-18 (exactly 18 Ry gates)
        for i in range(36, min(54, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = ((i - 36) % 18 + 1) % self.n_qubits  # Maps 36-53 to 1-18
            qml.RY(phi=angle, wires=wire_idx)
        
        # Features 55-60 → Rz rotations on qubits 1-6
        for i in range(54, min(60, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = ((i - 54) + 1) % self.n_qubits  # Maps 54-59 to 1-6
            qml.RZ(phi=angle, wires=wire_idx)

    def _encode_features_final_layer(self, x: np.ndarray) -> None:
        """Apply feature encoding for the final layer.
        
        Final encoding layer (20 features):
        * Features 61-64 → Rx rotations on qubits 7-10
        * Features 65-80 → Ry rotations on qubits 1-16 (exactly 16 Ry gates)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 61-64 → Rx rotations on qubits 7-10
        for i in range(60, min(64, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = ((i - 60) + 7) % self.n_qubits  # Maps 60-63 to 7-10
            qml.RX(phi=angle, wires=wire_idx)
        
        # Features 65-80 → Ry rotations on qubits 1-16 (exactly 16 Ry gates)
        for i in range(64, min(80, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = ((i - 64) % 16 + 1) % self.n_qubits  # Maps 64-79 to 1-16
            qml.RY(phi=angle, wires=wire_idx)

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
    
    def _apply_complementary_phase1(self) -> None:
        """Apply Complementary Phase pattern after Layer 1:
        Rz(π/3) to even-indexed qubits and Rz(π/2) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_complementary_phase2(self) -> None:
        """Apply Complementary Phase pattern after Layer 2:
        Rz(π/2) to even-indexed qubits and Rz(π/3) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_phase3(self) -> None:
        """Apply Phase pattern after Layer 3:
        Rz(π/4) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3_all, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_fine_tuned_hadamard_pattern(self) -> None:
        """Apply Fine-Tuned Fourier-Inspired Hadamard Pattern:
        - Qubit index mod 4 = 0: Apply H gate
        - Qubit index mod 4 = 1: Apply Rz(π/5) followed by H gate
        - Qubit index mod 4 = 2: Apply Rz(π/2) followed by H gate
        - Qubit index mod 4 = 3: Apply Rz(4π/5) followed by H gate
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
        # Hybrid repetition structure
        for rep in range(self.reps):
            # Encode features
            if rep == 0:
                self._encode_features_first_rep(x)
            else:
                self._encode_features_second_rep(x)
            
            # Apply entanglement layers with Complementary Phase pattern
            self._apply_local_entanglement()
            self._apply_complementary_phase1()
            
            self._apply_medium_entanglement()
            self._apply_complementary_phase2()
            
            self._apply_global_entanglement()
            self._apply_phase3()
            
            # Apply controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Apply final encoding layer
        self._encode_features_final_layer(x)
        
        # Apply Fine-Tuned Fourier-Inspired Hadamard Pattern
        self._apply_fine_tuned_hadamard_pattern()