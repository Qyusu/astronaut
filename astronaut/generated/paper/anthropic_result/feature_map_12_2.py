import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class FourierEnhancedModuloHadamardEncoder(BaseFeatureMap):
    """Fourier-Enhanced Modulo-4 Hadamard Pattern Encoder feature map.
    
    This feature map optimizes the final Hadamard layer using Fourier-inspired phase angles
    to maximize class distinguishability while maintaining effective elements from previous trials.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = FourierEnhancedModuloHadamardEncoder(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.92 * np.pi, 
        offset: float = np.pi / 3.5,
        phase1_even: float = np.pi / 6,
        phase1_odd: float = np.pi / 3,
        phase2_even: float = np.pi / 3,
        phase2_odd: float = np.pi / 6,
        phase3_all: float = np.pi / 2,
        fourier_phase1: float = np.pi / 4,
        fourier_phase2: float = np.pi / 2,
        fourier_phase3: float = 3 * np.pi / 4,
        reps: int = 2
    ) -> None:
        """Initialize the Fourier-Enhanced Modulo-4 Hadamard Pattern Encoder.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.92*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.5.
            phase1_even (float, optional): Phase for even qubits after layer 1. Defaults to np.pi/6.
            phase1_odd (float, optional): Phase for odd qubits after layer 1. Defaults to np.pi/3.
            phase2_even (float, optional): Phase for even qubits after layer 2. Defaults to np.pi/3.
            phase2_odd (float, optional): Phase for odd qubits after layer 2. Defaults to np.pi/6.
            phase3_all (float, optional): Phase for all qubits after layer 3. Defaults to np.pi/2.
            fourier_phase1 (float, optional): Fourier phase angle 1. Defaults to np.pi/4.
            fourier_phase2 (float, optional): Fourier phase angle 2. Defaults to np.pi/2.
            fourier_phase3 (float, optional): Fourier phase angle 3. Defaults to 3*np.pi/4.
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
        self.fourier_phase1: float = fourier_phase1
        self.fourier_phase2: float = fourier_phase2
        self.fourier_phase3: float = fourier_phase3
        self.reps: int = reps
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features_layer1(self, x: np.ndarray) -> None:
        """Apply the first layer of feature encoding.
        
        Features 1-10 → Ry rotations on qubits 1-10
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        for i in range(min(10, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=i % self.n_qubits)

    def _encode_features_layer2(self, x: np.ndarray) -> None:
        """Apply the second layer of feature encoding.
        
        Features 11-20 → Ry rotations on qubits 1-10 (second layer)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        for i in range(10, min(20, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 10) % self.n_qubits)

    def _encode_features_layer3(self, x: np.ndarray) -> None:
        """Apply the third layer of feature encoding.
        
        Features 21-30 → Ry rotations on qubits 1-10 (third layer)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        for i in range(20, min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 20) % self.n_qubits)

    def _encode_features_layer4(self, x: np.ndarray) -> None:
        """Apply the fourth layer of feature encoding.
        
        Features 31-40 → Rx rotations on qubits 1-10
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        for i in range(30, min(40, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 30) % self.n_qubits)

    def _encode_features_layer5(self, x: np.ndarray) -> None:
        """Apply the fifth layer of feature encoding.
        
        Features 41-50 → Rx rotations on qubits 1-10 (second layer)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        for i in range(40, min(50, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 40) % self.n_qubits)

    def _encode_features_layer6(self, x: np.ndarray) -> None:
        """Apply the sixth layer of feature encoding.
        
        Features 51-62 → Ry rotations on qubits 1-10 (partial fourth layer)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        for i in range(50, min(62, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 50) % self.n_qubits)

    def _encode_features_layer7(self, x: np.ndarray) -> None:
        """Apply the seventh layer of feature encoding.
        
        Features 63-70 → Rz rotations on qubits 1-8 (partial first layer)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        for i in range(62, min(70, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 62) % self.n_qubits)

    def _encode_features_layer8(self, x: np.ndarray) -> None:
        """Apply the eighth layer of feature encoding.
        
        Features 71-80 → Ry rotations on qubits 1-10 (partial fifth layer)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        for i in range(70, min(80, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 70) % self.n_qubits)

    def _encode_features_first_rep(self, x: np.ndarray) -> None:
        """Apply feature encoding layers for the first repetition.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        self._encode_features_layer1(x)
        self._encode_features_layer2(x)
        self._encode_features_layer3(x)
        self._encode_features_layer4(x)
        
    def _encode_features_second_rep(self, x: np.ndarray) -> None:
        """Apply feature encoding layers for the second repetition.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        self._encode_features_layer5(x)
        self._encode_features_layer6(x)
        self._encode_features_layer7(x)
        self._encode_features_layer8(x)

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
        Rz(π/6) to even-indexed qubits and Rz(π/3) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_phase_pattern2(self) -> None:
        """Apply Phase pattern after Layer 2:
        Rz(π/3) to even-indexed qubits and Rz(π/6) to odd-indexed qubits."""
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
    
    def _apply_fourier_hadamard_pattern(self) -> None:
        """Apply Fourier-optimized Hadamard pattern based on qubit index modulo 4:
        - Qubit index mod 4 = 0: Apply H gate
        - Qubit index mod 4 = 1: Apply Rz(π/4) followed by H gate
        - Qubit index mod 4 = 2: Apply Rz(π/2) followed by H gate
        - Qubit index mod 4 = 3: Apply Rz(3π/4) followed by H gate
        """
        for i in range(self.n_qubits):
            mod4 = i % 4
            if mod4 == 0:
                qml.Hadamard(wires=i)
            elif mod4 == 1:
                qml.RZ(phi=self.fourier_phase1, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 2:
                qml.RZ(phi=self.fourier_phase2, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 3:
                qml.RZ(phi=self.fourier_phase3, wires=i)
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
            
            # Apply entanglement layers with Phase pattern
            self._apply_local_entanglement()
            self._apply_phase_pattern1()
            
            self._apply_medium_entanglement()
            self._apply_phase_pattern2()
            
            self._apply_global_entanglement()
            self._apply_phase_pattern3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Apply Fourier-optimized modulo-4 Hadamard pattern
        self._apply_fourier_hadamard_pattern()