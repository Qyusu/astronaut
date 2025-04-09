import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class OptimallyBalancedRyEncoder(BaseFeatureMap):
    """Optimally Balanced Ry Encoder feature map.
    
    Implements an evenly distributed proportion of Ry gates across all repetitions
    while maintaining the effective elements from previous trials.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = OptimallyBalancedRyEncoder(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.92 * np.pi, 
        offset: float = np.pi / 3.5,
        phase1_even: float = np.pi / 8,
        phase1_odd: float = np.pi / 4,
        phase2_even: float = np.pi / 4,
        phase2_odd: float = np.pi / 2,
        phase3_all: float = np.pi / 2,
        reps: int = 2
    ) -> None:
        """Initialize the Optimally Balanced Ry Encoder.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.92*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.5.
            phase1_even (float, optional): Phase for even qubits after layer 1. Defaults to np.pi/8.
            phase1_odd (float, optional): Phase for odd qubits after layer 1. Defaults to np.pi/4.
            phase2_even (float, optional): Phase for even qubits after layer 2. Defaults to np.pi/4.
            phase2_odd (float, optional): Phase for odd qubits after layer 2. Defaults to np.pi/2.
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

    def _encode_features_rep1(self, x: np.ndarray) -> None:
        """Apply feature encoding for the first repetition.
        
        First repetition (30 features):
        * Features 1-6 → Rx rotations on qubits 1-6
        * Features 7-26 → Ry rotations on qubits 7-10 and 1-10 (twice)
        * Features 27-30 → Rz rotations on qubits 7-10
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 1-6 → Rx rotations on qubits 1-6
        for i in range(min(6, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 7-26 → Ry rotations on qubits 7-10 and 1-10 (twice)
        for i in range(6, min(26, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map to qubits 7-10 first, then 1-10 twice (if needed)
            qubit_idx = (i - 6) % self.n_qubits
            if i < 10:  # First 4 features go to qubits 7-10
                qubit_idx = (qubit_idx + 6) % self.n_qubits
            qml.RY(phi=angle, wires=qubit_idx)
        
        # Features 27-30 → Rz rotations on qubits 7-10
        for i in range(26, min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qubit_idx = (i - 26 + 6) % self.n_qubits  # Maps to qubits 7-10
            qml.RZ(phi=angle, wires=qubit_idx)

    def _encode_features_rep2(self, x: np.ndarray) -> None:
        """Apply feature encoding for the second repetition.
        
        Second repetition (30 features):
        * Features 31-36 → Rx rotations on qubits 7-10 and 1-2
        * Features 37-56 → Ry rotations on qubits 3-10 and 1-10 (twice)
        * Features 57-60 → Rz rotations on qubits 3-6
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 31-36 → Rx rotations on qubits 7-10 and 1-2
        for i in range(30, min(36, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map features to qubits 7-10 first, then 1-2
            feature_idx = i - 30
            if feature_idx < 4:  # First 4 go to qubits 7-10
                qubit_idx = feature_idx + 6
            else:  # Next 2 go to qubits 1-2
                qubit_idx = feature_idx - 4
            qml.RX(phi=angle, wires=qubit_idx % self.n_qubits)
        
        # Features 37-56 → Ry rotations on qubits 3-10 and 1-10 (twice)
        for i in range(36, min(56, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map to qubits 3-10 first, then 1-10
            feature_idx = i - 36
            if feature_idx < 8:  # First 8 go to qubits 3-10
                qubit_idx = feature_idx + 2
            else:  # Remaining go to qubits 1-10
                qubit_idx = feature_idx - 8
            qml.RY(phi=angle, wires=qubit_idx % self.n_qubits)
        
        # Features 57-60 → Rz rotations on qubits 3-6
        for i in range(56, min(60, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qubit_idx = i - 56 + 2  # Maps to qubits 3-6
            qml.RZ(phi=angle, wires=qubit_idx % self.n_qubits)

    def _encode_features_final(self, x: np.ndarray) -> None:
        """Apply feature encoding for the final layer.
        
        Final encoding layer (20 features):
        * Features 61-68 → Rx rotations on qubits 7-10 and 1-4
        * Features 69-80 → Ry rotations on qubits 5-10 and 1-6
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 61-68 → Rx rotations on qubits 7-10 and 1-4
        for i in range(60, min(68, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map features to qubits 7-10 first, then 1-4
            feature_idx = i - 60
            if feature_idx < 4:  # First 4 go to qubits 7-10
                qubit_idx = feature_idx + 6
            else:  # Next 4 go to qubits 1-4
                qubit_idx = feature_idx - 4
            qml.RX(phi=angle, wires=qubit_idx % self.n_qubits)
        
        # Features 69-80 → Ry rotations on qubits 5-10 and 1-6
        for i in range(68, min(80, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map to qubits 5-10 first, then 1-6
            feature_idx = i - 68
            if feature_idx < 6:  # First 6 go to qubits 5-10
                qubit_idx = feature_idx + 4
            else:  # Remaining go to qubits 1-6
                qubit_idx = feature_idx - 6
            qml.RY(phi=angle, wires=qubit_idx % self.n_qubits)

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
    
    def _apply_phase_pattern1(self) -> None:
        """Apply Phase pattern after Layer 1:
        Rz(π/8) to even-indexed qubits and Rz(π/4) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_phase_pattern2(self) -> None:
        """Apply Phase pattern after Layer 2:
        Rz(π/4) to even-indexed qubits and Rz(π/2) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_phase_pattern3(self) -> None:
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
    
    def _apply_mod4_hadamard(self) -> None:
        """Apply systematic Hadamard pattern based on qubit index modulo 4:
        - Qubit index mod 4 = 0: Apply H gate
        - Qubit index mod 4 = 1: Apply Rz(π/6) followed by H gate
        - Qubit index mod 4 = 2: Apply Rz(π/4) followed by H gate
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
                qml.RZ(phi=np.pi/4, wires=i)
                qml.Hadamard(wires=i)
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
        rep_count = 0
        for _ in range(self.reps):
            # Encode features with Ry-prioritized distribution
            if rep_count == 0:
                self._encode_features_rep1(x)
            else:
                self._encode_features_rep2(x)
            
            # Apply entanglement layers with Phase pattern
            self._apply_local_entanglement()
            self._apply_phase_pattern1()
            
            self._apply_medium_entanglement()
            self._apply_phase_pattern2()
            
            self._apply_global_entanglement()
            self._apply_phase_pattern3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
            
            rep_count += 1
        
        # Additional feature encoding layer
        self._encode_features_final(x)
        
        # Apply systematic modulo-4 Hadamard pattern
        self._apply_mod4_hadamard()