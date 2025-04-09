import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class TargetedGlobalConnectivityWithFibonacciPhaseHarmony(BaseFeatureMap):
    """Targeted Global Connectivity with Fibonacci Phase Harmony feature map.
    
    A quantum feature map that implements specialized entanglement connections
    and mathematically harmonious phase relationships.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = TargetedGlobalConnectivityWithFibonacciPhaseHarmony(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.9 * np.pi, 
        offset: float = np.pi / 3,
        phase1_even: float = np.pi / 5,
        phase1_odd: float = np.pi / 3,
        phase2_even: float = np.pi / 3,
        phase2_odd: float = np.pi / 2,
        phase3: float = 5 * np.pi / 6,
        golden_ratio_conjugate: float = 0.382,
        golden_ratio: float = 0.618,
        reps: int = 2
    ) -> None:
        """Initialize the Targeted Global Connectivity feature map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.9*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.
            phase1_even (float, optional): Phase shift for even qubits after layer 1. Defaults to np.pi/5.
            phase1_odd (float, optional): Phase shift for odd qubits after layer 1. Defaults to np.pi/3.
            phase2_even (float, optional): Phase shift for even qubits after layer 2. Defaults to np.pi/3.
            phase2_odd (float, optional): Phase shift for odd qubits after layer 2. Defaults to np.pi/2.
            phase3 (float, optional): Phase shift for all qubits after layer 3. Defaults to 5*np.pi/6.
            golden_ratio_conjugate (float, optional): Golden ratio conjugate. Defaults to 0.382.
            golden_ratio (float, optional): Golden ratio. Defaults to 0.618.
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
        self.golden_ratio_conjugate: float = golden_ratio_conjugate
        self.golden_ratio: float = golden_ratio
        self.reps: int = reps
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with high proportion of Ry gates (65%).
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 1-10 → Ry rotations on qubits 1-10
        for i in range(min(10, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=i % self.n_qubits)
        
        # Features 11-20 → Ry rotations on qubits 1-10 (second layer)
        for i in range(10, min(20, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 10) % self.n_qubits)
        
        # Features 21-30 → Ry rotations on qubits 1-10 (third layer)
        for i in range(20, min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 20) % self.n_qubits)
        
        # Features 31-40 → Rx rotations on qubits 1-10
        for i in range(30, min(40, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 30) % self.n_qubits)
        
        # Features 41-50 → Rx rotations on qubits 1-10 (second layer)
        for i in range(40, min(50, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 40) % self.n_qubits)
        
        # Features 51-62 → Ry rotations on qubits 1-10 (partial fourth layer)
        for i in range(50, min(62, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 50) % self.n_qubits)
        
        # Features 63-80 → Rz rotations on qubits 1-10 (partial first & second layer)
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
    
    def _apply_targeted_global_entanglement(self) -> None:
        """Apply targeted global connections based on qubit index parity (Layer 3).
        
        For even-indexed qubits: Connect qubit i to qubit (i+3) mod n
        For odd-indexed qubits: Connect qubit i to qubit (i+4) mod n
        """
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.CNOT(wires=[i, (i + 3) % self.n_qubits])
            else:  # odd
                qml.CNOT(wires=[i, (i + 4) % self.n_qubits])
    
    def _apply_fibonacci_phase1(self) -> None:
        """Apply Fibonacci-enhanced phase structure after Layer 1:
        Rz(π/5) to even-indexed qubits and Rz(π/3) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_fibonacci_phase2(self) -> None:
        """Apply Fibonacci-enhanced phase structure after Layer 2:
        Rz(π/3) to even-indexed qubits and Rz(π/2) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_fibonacci_phase3(self) -> None:
        """Apply Fibonacci-enhanced phase structure after Layer 3:
        Rz(5π/6) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_golden_ratio_hadamard(self) -> None:
        """Apply golden ratio-based Hadamard pattern based on qubit index modulo 4:
        - Qubit index mod 4 = 0: Apply H gate
        - Qubit index mod 4 = 1: Apply Rz(π·0.382) followed by H gate
        - Qubit index mod 4 = 2: Apply Rz(π·0.618) followed by H gate
        - Qubit index mod 4 = 3: Apply Rz(π) followed by H gate
        """
        for i in range(self.n_qubits):
            mod4 = i % 4
            if mod4 == 0:
                qml.Hadamard(wires=i)
            elif mod4 == 1:
                qml.RZ(phi=np.pi * self.golden_ratio_conjugate, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 2:
                qml.RZ(phi=np.pi * self.golden_ratio, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 3:
                qml.RZ(phi=np.pi, wires=i)
                qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Hybrid repetition structure
        for _ in range(self.reps):
            # Encode features with high proportion of Ry gates
            self._encode_features(x)
            
            # Apply entanglement layers with Fibonacci-enhanced phase structure
            self._apply_local_entanglement()
            self._apply_fibonacci_phase1()
            
            self._apply_medium_entanglement()
            self._apply_fibonacci_phase2()
            
            self._apply_targeted_global_entanglement()
            self._apply_fibonacci_phase3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Additional feature encoding layer
        self._encode_features(x)
        
        # Apply golden ratio modulo-4 Hadamard pattern
        self._apply_golden_ratio_hadamard()