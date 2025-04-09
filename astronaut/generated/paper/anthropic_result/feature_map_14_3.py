import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class ExtendedFourierEncoder(BaseFeatureMap):
    """Extended Fourier Quantum Encoder with Perfect Balance.

    This feature map implements a comprehensive Fourier-inspired design with 
    perfectly balanced parameters and golden ratio relationships.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = ExtendedFourierEncoder(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.94 * np.pi, 
        offset: float = np.pi / 3.35,
        golden_ratio: float = 1.618,
        reps: int = 2
    ) -> None:
        """Initialize the Extended Fourier Quantum Encoder feature map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.94*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.35.
            golden_ratio (float, optional): Golden ratio value. Defaults to 1.618.
            reps (int, optional): Number of repetitions. Defaults to 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        self.golden_ratio: float = golden_ratio
        self.reps: int = reps
        
        # Precompute the phase angles based on golden ratio
        self.phase1_even = np.pi / (1 + golden_ratio)  # ≈ 0.382π
        self.phase1_odd = np.pi / golden_ratio  # ≈ 0.618π
        self.phase2_even = np.pi / golden_ratio  # ≈ 0.618π
        self.phase2_odd = np.pi / (1 + golden_ratio)  # ≈ 0.382π
        self.phase3_all = np.pi / 2  # 0.5π
        
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
        # Features 1-6 → Rx rotations on qubits 1-6 (0-5 in 0-indexed)
        for i in range(min(6, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 7-24 → Ry rotations on qubits 7-10 and 1-14 (exactly 18 Ry gates)
        for i in range(6, min(24, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map to qubits 7-10 (6-9 in 0-indexed) and then 1-14 (0-13 in 0-indexed)
            if i < 10:  # For features 7-10 map to qubits 7-10 (6-9 in 0-indexed)
                wire_idx = i
            else:  # For features 11-24 map to qubits 1-14 (0-13 in 0-indexed)
                wire_idx = (i - 10) % self.n_qubits
            qml.RY(phi=angle, wires=wire_idx % self.n_qubits)
        
        # Features 25-30 → Rz rotations on qubits 5-10 (4-9 in 0-indexed)
        for i in range(24, min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 24 + 4) % self.n_qubits  # Map to qubits 5-10 (4-9 in 0-indexed)
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
        # Features 31-36 → Rx rotations on qubits 5-10 (4-9 in 0-indexed)
        for i in range(30, min(36, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 30 + 4) % self.n_qubits  # Map to qubits 5-10 (4-9 in 0-indexed)
            qml.RX(phi=angle, wires=wire_idx)
        
        # Features 37-54 → Ry rotations on qubits 1-18 (0-17 in 0-indexed)
        for i in range(36, min(54, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 36) % self.n_qubits  # Map to qubits 1-18 (0-17 in 0-indexed)
            qml.RY(phi=angle, wires=wire_idx)
        
        # Features 55-60 → Rz rotations on qubits 1-6 (0-5 in 0-indexed)
        for i in range(54, min(60, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 54) % self.n_qubits  # Map to qubits 1-6 (0-5 in 0-indexed)
            qml.RZ(phi=angle, wires=wire_idx)

    def _encode_final_layer(self, x: np.ndarray) -> None:
        """Apply feature encoding for the final layer.
        
        Final encoding layer (20 features):
        * Features 61-64 → Rx rotations on qubits 7-10
        * Features 65-80 → Ry rotations on qubits 1-16 (exactly 16 Ry gates)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 61-64 → Rx rotations on qubits 7-10 (6-9 in 0-indexed)
        for i in range(60, min(64, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 60 + 6) % self.n_qubits  # Map to qubits 7-10 (6-9 in 0-indexed)
            qml.RX(phi=angle, wires=wire_idx)
        
        # Features 65-80 → Ry rotations on qubits 1-16 (0-15 in 0-indexed)
        for i in range(64, min(80, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 64) % self.n_qubits  # Map to qubits 1-16 (0-15 in 0-indexed)
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

    def _apply_phase1_golden(self) -> None:
        """Apply Phase pattern after Layer 1:
        Rz(π/(1+φ)) ≈ Rz(0.382π) to even-indexed qubits and 
        Rz(π/φ) ≈ Rz(0.618π) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)

    def _apply_phase2_golden(self) -> None:
        """Apply Phase pattern after Layer 2:
        Rz(π/φ) ≈ Rz(0.618π) to even-indexed qubits and
        Rz(π/(1+φ)) ≈ Rz(0.382π) to odd-indexed qubits."""
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

    def _apply_extended_hadamard(self) -> None:
        """Apply Extended Modulo-8 Hadamard Pattern:
        - Qubit index mod 8 = 0: Apply H gate
        - Qubit index mod 8 = 1: Apply Rz(π/8) followed by H gate
        - Qubit index mod 8 = 2: Apply Rz(π/4) followed by H gate
        - Qubit index mod 8 = 3: Apply Rz(3π/8) followed by H gate
        - Qubit index mod 8 = 4: Apply Rz(π/2) followed by H gate
        - Qubit index mod 8 = 5: Apply Rz(5π/8) followed by H gate
        - Qubit index mod 8 = 6: Apply Rz(3π/4) followed by H gate
        - Qubit index mod 8 = 7: Apply Rz(7π/8) followed by H gate
        """
        for i in range(self.n_qubits):
            mod8 = i % 8
            if mod8 == 0:
                qml.Hadamard(wires=i)
            else:
                # Apply Rz with angle (mod8 * π / 8)
                qml.RZ(phi=(mod8 * np.pi / 8), wires=i)
                qml.Hadamard(wires=i)

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Apply feature map structure for specified number of repetitions
        for rep in range(self.reps):
            # Apply appropriate feature encoding for this repetition
            if rep == 0:
                self._encode_features_first_rep(x)
            else:
                self._encode_features_second_rep(x)
            
            # Apply entanglement layers with golden ratio phase patterns
            self._apply_local_entanglement()
            self._apply_phase1_golden()
            
            self._apply_medium_entanglement()
            self._apply_phase2_golden()
            
            self._apply_global_entanglement()
            self._apply_phase3()
            
            # Apply controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Apply final encoding layer
        self._encode_final_layer(x)
        
        # Apply extended modulo-8 Hadamard pattern
        self._apply_extended_hadamard()