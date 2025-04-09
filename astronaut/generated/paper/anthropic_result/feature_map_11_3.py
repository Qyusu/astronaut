import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class PhaseHarmonyEncoderWithBalancedRyPrioritization(BaseFeatureMap):
    """Phase Harmony Encoder with Balanced Ry Prioritization feature map.
    
    A quantum feature map that creates a balanced distribution with Ry gates prioritized 
    in earlier layers and implements harmonious phase relationships.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = PhaseHarmonyEncoderWithBalancedRyPrioritization(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.9 * np.pi, 
        offset: float = np.pi / 3,
        phase1_even: float = np.pi / 4,
        phase1_odd: float = np.pi / 2,
        phase2_even: float = np.pi / np.sqrt(2),
        phase2_odd: float = np.pi / np.sqrt(3),
        phase3_all: float = np.pi / 2,
        reps: int = 2
    ) -> None:
        """Initialize the Phase Harmony Encoder with Balanced Ry Prioritization.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.9*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.
            phase1_even (float, optional): Phase for even qubits after layer 1. Defaults to np.pi/4.
            phase1_odd (float, optional): Phase for odd qubits after layer 1. Defaults to np.pi/2.
            phase2_even (float, optional): Phase for even qubits after layer 2. Defaults to np.pi/sqrt(2).
            phase2_odd (float, optional): Phase for odd qubits after layer 2. Defaults to np.pi/sqrt(3).
            phase3_all (float, optional): Phase for all qubits after layer 3. Defaults to np.pi/2.
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
        self.reps: int = reps
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with Ry gates prioritized in the first half of the circuit.
        
        First 40 features (first half):
          - Features 1-30 → Ry rotations on qubits 1-10 (3 complete layers)
          - Features 31-40 → Rx rotations on qubits 1-10
        Second 40 features (second half):
          - Features 41-50 → Rx rotations on qubits 1-10
          - Features 51-62 → Ry rotations on qubits 1-10 (partial layer)
          - Features 63-80 → Rz rotations on qubits 1-10 (partial layers)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 1-30 → Ry rotations on qubits 1-10 (3 complete layers)
        for i in range(min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=i % self.n_qubits)
        
        # Features 31-40 → Rx rotations on qubits 1-10
        for i in range(30, min(40, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 30) % self.n_qubits)
        
        # Features 41-50 → Rx rotations on qubits 1-10
        for i in range(40, min(50, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 40) % self.n_qubits)
        
        # Features 51-62 → Ry rotations on qubits 1-10 (partial layer)
        for i in range(50, min(62, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 50) % self.n_qubits)
        
        # Features 63-80 → Rz rotations on qubits 1-10 (partial layers)
        for i in range(62, min(80, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 62) % self.n_qubits)

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
        # For a 10-qubit system, this connects qubits with distance 3
        distance = max(1, self.n_qubits // 3)
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + distance) % self.n_qubits])
    
    def _apply_phase_harmony1(self) -> None:
        """Apply Phase Harmony structure after Layer 1:
        Rz(π/4) to even-indexed qubits and Rz(π/2) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_phase_harmony2(self) -> None:
        """Apply Phase Harmony structure after Layer 2:
        Rz(π/√2) to even-indexed qubits and Rz(π/√3) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_phase_harmony3(self) -> None:
        """Apply Phase Harmony structure after Layer 3:
        Rz(π/√4) = Rz(π/2) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3_all, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_optimized_hadamard(self) -> None:
        """Apply optimized Hadamard pattern based on qubit index modulo 4:
        - Qubit index mod 4 = 0: Apply H gate
        - Qubit index mod 4 = 1: Apply Rz(π/6) followed by H gate
        - Qubit index mod 4 = 2: Apply H gate followed by Rz(π/6)
        - Qubit index mod 4 = 3: Apply Rz(π/3) followed by H gate
        """
        for i in range(self.n_qubits):
            mod4 = i % 4
            if mod4 == 0:
                qml.Hadamard(wires=i)
            elif mod4 == 1:
                qml.RZ(phi=np.pi/6, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 2:
                qml.Hadamard(wires=i)
                qml.RZ(phi=np.pi/6, wires=i)
            elif mod4 == 3:
                qml.RZ(phi=np.pi/3, wires=i)
                qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Hybrid repetition structure
        for _ in range(self.reps):
            # Encode features with Ry-prioritized distribution
            self._encode_features(x)
            
            # Apply entanglement layers with Phase Harmony structure
            self._apply_local_entanglement()
            self._apply_phase_harmony1()
            
            self._apply_medium_entanglement()
            self._apply_phase_harmony2()
            
            self._apply_global_entanglement()
            self._apply_phase_harmony3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Additional feature encoding layer
        self._encode_features(x)
        
        # Apply optimized modulo-4 Hadamard pattern
        self._apply_optimized_hadamard()