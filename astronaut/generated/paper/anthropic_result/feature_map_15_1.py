import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class EnhancedRyDualSequencesEncoder(BaseFeatureMap):
    """Enhanced Ry Quantum Encoder with Dual Mathematical Sequences.

    This feature map maximizes classification performance by increasing
    Ry gate proportion and implementing mathematically complementary
    phase patterns based on Fibonacci and Golden Ratio sequences.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = EnhancedRyDualSequencesEncoder(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.95 * np.pi, 
        offset: float = np.pi / 3.3,
        golden_ratio: float = 1.618,
        reps: int = 2
    ) -> None:
        """Initialize the Enhanced Ry Dual Sequences Encoder feature map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.95*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.3.
            golden_ratio (float, optional): Golden ratio value for phase calculations. Defaults to 1.618.
            reps (int, optional): Number of repetitions. Defaults to 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        self.golden_ratio: float = golden_ratio
        self.reps: int = reps
        
        # Precompute the phase angles
        # Fibonacci-derived phases
        self.fib_phase1_even = np.pi / 5
        self.fib_phase1_odd = np.pi / 3
        self.fib_phase2_even = np.pi / 3
        self.fib_phase2_odd = np.pi / 2
        self.fib_phase3_all = np.pi / 8
        
        # Golden ratio phases
        self.golden_phase1_even = np.pi / (1 + golden_ratio)  # ≈ 0.382π
        self.golden_phase1_odd = np.pi / golden_ratio  # ≈ 0.618π
        self.golden_phase2_even = np.pi / golden_ratio  # ≈ 0.618π
        self.golden_phase2_odd = np.pi / (1 + golden_ratio)  # ≈ 0.382π
        self.golden_phase3_all = np.pi / 2  # 0.5π
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features_first_rep(self, x: np.ndarray) -> None:
        """Apply feature encoding for the first repetition.
        
        First repetition (30 features):
        * Features 1-6 → Rx rotations on qubits 1-6
        * Features 7-25 → Ry rotations on qubits 7-10 and 1-15 (exactly 19 Ry gates)
        * Features 26-30 → Rz rotations on qubits 6-10
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 1-6 → Rx rotations on qubits 1-6 (0-5 in 0-indexed)
        for i in range(min(6, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 7-25 → Ry rotations on qubits 7-10 and 1-15 (exactly 19 Ry gates)
        for i in range(6, min(25, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map to qubits 7-10 (6-9 in 0-indexed) and then 1-15 (0-14 in 0-indexed)
            if i < 10:  # For features 7-10 map to qubits 7-10 (6-9 in 0-indexed)
                wire_idx = i
            else:  # For features 11-25 map to qubits 1-15 (0-14 in 0-indexed)
                wire_idx = (i - 10) % self.n_qubits
            qml.RY(phi=angle, wires=wire_idx % self.n_qubits)
        
        # Features 26-30 → Rz rotations on qubits 6-10 (5-9 in 0-indexed)
        for i in range(25, min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 25 + 5) % self.n_qubits  # Map to qubits 6-10 (5-9 in 0-indexed)
            qml.RZ(phi=angle, wires=wire_idx)

    def _encode_features_second_rep(self, x: np.ndarray) -> None:
        """Apply feature encoding for the second repetition.
        
        Second repetition (30 features):
        * Features 31-36 → Rx rotations on qubits 5-10
        * Features 37-55 → Ry rotations on qubits 1-19 (exactly 19 Ry gates)
        * Features 56-60 → Rz rotations on qubits 1-5
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 31-36 → Rx rotations on qubits 5-10 (4-9 in 0-indexed)
        for i in range(30, min(36, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 30 + 4) % self.n_qubits  # Map to qubits 5-10 (4-9 in 0-indexed)
            qml.RX(phi=angle, wires=wire_idx)
        
        # Features 37-55 → Ry rotations on qubits 1-19 (0-18 in 0-indexed)
        for i in range(36, min(55, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 36) % self.n_qubits  # Map to qubits 1-19 (0-18 in 0-indexed)
            qml.RY(phi=angle, wires=wire_idx)
        
        # Features 56-60 → Rz rotations on qubits 1-5 (0-4 in 0-indexed)
        for i in range(55, min(60, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            wire_idx = (i - 55) % self.n_qubits  # Map to qubits 1-5 (0-4 in 0-indexed)
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

    def _apply_fibonacci_phase1(self) -> None:
        """Apply Fibonacci-derived Phase pattern after Layer 1:
        Rz(π/5) to even-indexed qubits and 
        Rz(π/3) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.fib_phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.fib_phase1_odd, wires=i)

    def _apply_fibonacci_phase2(self) -> None:
        """Apply Fibonacci-derived Phase pattern after Layer 2:
        Rz(π/3) to even-indexed qubits and
        Rz(π/2) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.fib_phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.fib_phase2_odd, wires=i)

    def _apply_fibonacci_phase3(self) -> None:
        """Apply Fibonacci-derived Phase pattern after Layer 3:
        Rz(π/8) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.fib_phase3_all, wires=i)

    def _apply_golden_phase1(self) -> None:
        """Apply Golden ratio Phase pattern after Layer 1:
        Rz(π/(1+φ)) ≈ Rz(0.382π) to even-indexed qubits and 
        Rz(π/φ) ≈ Rz(0.618π) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.golden_phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.golden_phase1_odd, wires=i)

    def _apply_golden_phase2(self) -> None:
        """Apply Golden ratio Phase pattern after Layer 2:
        Rz(π/φ) ≈ Rz(0.618π) to even-indexed qubits and
        Rz(π/(1+φ)) ≈ Rz(0.382π) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.golden_phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.golden_phase2_odd, wires=i)

    def _apply_golden_phase3(self) -> None:
        """Apply Golden ratio Phase pattern after Layer 3:
        Rz(π/2) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.golden_phase3_all, wires=i)

    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])

    def _apply_optimized_hadamard(self) -> None:
        """Apply Optimized Hadamard Pattern:
        - Qubit index mod 4 = 0: Apply H gate
        - Qubit index mod 4 = 1: Apply Rz(π/7) followed by H gate
        - Qubit index mod 4 = 2: Apply Rz(π/2) followed by H gate
        - Qubit index mod 4 = 3: Apply Rz(6π/7) followed by H gate
        """
        for i in range(self.n_qubits):
            mod4 = i % 4
            if mod4 == 0:
                qml.Hadamard(wires=i)
            elif mod4 == 1:
                qml.RZ(phi=np.pi / 7, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 2:
                qml.RZ(phi=np.pi / 2, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 3:
                qml.RZ(phi=6 * np.pi / 7, wires=i)
                qml.Hadamard(wires=i)

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # First repetition with Fibonacci-derived phases
        self._encode_features_first_rep(x)
        
        self._apply_local_entanglement()
        self._apply_fibonacci_phase1()
        
        self._apply_medium_entanglement()
        self._apply_fibonacci_phase2()
        
        self._apply_global_entanglement()
        self._apply_fibonacci_phase3()
        
        self._apply_cz_triplets()
        
        # Second repetition with Golden ratio phases
        self._encode_features_second_rep(x)
        
        self._apply_local_entanglement()
        self._apply_golden_phase1()
        
        self._apply_medium_entanglement()
        self._apply_golden_phase2()
        
        self._apply_global_entanglement()
        self._apply_golden_phase3()
        
        self._apply_cz_triplets()
        
        # Final encoding layer
        self._encode_final_layer(x)
        
        # Apply optimized Hadamard pattern
        self._apply_optimized_hadamard()