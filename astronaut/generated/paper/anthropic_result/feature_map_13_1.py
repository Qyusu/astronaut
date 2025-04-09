import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class ExponentialFourierHarmonicSynthesisEncoder(BaseFeatureMap):
    """Exponential Fourier-Harmonic Synthesis Encoder feature map.
    
    This feature map enhances the original design by incorporating exponential frequency 
    growth principles from recent research on quantum circuit expressivity.
    
    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = ExponentialFourierHarmonicSynthesisEncoder(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.92 * np.pi, 
        offset: float = np.pi / 3.5,
        phase1_even: float = np.pi / 2,
        phase1_odd: float = np.pi / 4,
        phase2_even: float = np.pi / 8,
        phase2_odd: float = np.pi / 16,
        phase3_all: float = np.pi / 32,
        h_mod_phase1: float = np.pi / 4,
        h_mod_phase2: float = np.pi / 2,
        h_mod_phase3: float = 3 * np.pi / 4,
        reps: int = 2
    ) -> None:
        """Initialize the Exponential Fourier-Harmonic Synthesis Encoder.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.92*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.5.
            phase1_even (float, optional): Phase for even qubits after layer 1. Defaults to np.pi/2.
            phase1_odd (float, optional): Phase for odd qubits after layer 1. Defaults to np.pi/4.
            phase2_even (float, optional): Phase for even qubits after layer 2. Defaults to np.pi/8.
            phase2_odd (float, optional): Phase for odd qubits after layer 2. Defaults to np.pi/16.
            phase3_all (float, optional): Phase for all qubits after layer 3. Defaults to np.pi/32.
            h_mod_phase1 (float, optional): Phase for mod 4 = 1 qubits before Hadamard. Defaults to np.pi/4.
            h_mod_phase2 (float, optional): Phase for mod 4 = 2 qubits before Hadamard. Defaults to np.pi/2.
            h_mod_phase3 (float, optional): Phase for mod 4 = 3 qubits before Hadamard. Defaults to 3*np.pi/4.
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
        * Features 1-8 → Rx rotations on qubits 1-8
        * Features 9-26 → Ry rotations on qubits 9-10 and 1-16
        * Features 27-30 → Rz rotations on qubits 7-10
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 1-8 → Rx rotations on qubits 1-8
        for i in range(min(8, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 9-26 → Ry rotations on qubits 9-10 and 1-16
        for i in range(8, min(26, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            # Map to qubits 9-10 and then 1-16
            if i < 10:
                wire_idx = i % self.n_qubits
            else:
                wire_idx = (i - 10) % self.n_qubits
            qml.RY(phi=angle, wires=wire_idx)
        
        # Features 27-30 → Rz rotations on qubits 7-10
        for i in range(26, min(30, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 19) % self.n_qubits)  # Maps 26-29 to 7-10

    def _encode_features_second_rep(self, x: np.ndarray) -> None:
        """Apply feature encoding for the second repetition.
        
        Second repetition (30 features):
        * Features 31-38 → Rx rotations on qubits 3-10
        * Features 39-56 → Ry rotations on qubits 1-18
        * Features 57-60 → Rz rotations on qubits 7-10
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 31-38 → Rx rotations on qubits 3-10
        for i in range(30, min(38, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 27) % self.n_qubits)  # Maps 30-37 to 3-10
        
        # Features 39-56 → Ry rotations on qubits 1-18
        for i in range(38, min(56, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 38) % self.n_qubits)  # Maps 38-55 to 0-17 (mod 10)
        
        # Features 57-60 → Rz rotations on qubits 7-10
        for i in range(56, min(60, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 49) % self.n_qubits)  # Maps 56-59 to 7-10

    def _encode_features_final_layer(self, x: np.ndarray) -> None:
        """Apply feature encoding for the final layer.
        
        Final encoding layer (20 features):
        * Features 61-64 → Rx rotations on qubits 3-6
        * Features 65-80 → Ry rotations on qubits 7-10 and 1-12
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Features 61-64 → Rx rotations on qubits 3-6
        for i in range(60, min(64, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 57) % self.n_qubits)  # Maps 60-63 to 3-6
        
        # Features 65-80 → Ry rotations on qubits 7-10 and 1-12
        for i in range(64, min(80, len(x))):
            angle = self.scale_factor * x[i] + self.offset
            if i < 68:
                wire_idx = (i - 57) % self.n_qubits  # Maps 64-67 to 7-10
            else:
                wire_idx = (i - 68) % self.n_qubits  # Maps 68-79 to 0-11 (mod 10)
            qml.RY(phi=angle, wires=wire_idx)

    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_global_exponential_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 4 (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 4) % self.n_qubits])
    
    def _apply_exponential_phase1(self) -> None:
        """Apply Exponential Phase pattern after Layer 1:
        Rz(π/2) to even-indexed qubits and Rz(π/4) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_exponential_phase2(self) -> None:
        """Apply Exponential Phase pattern after Layer 2:
        Rz(π/8) to even-indexed qubits and Rz(π/16) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_exponential_phase3(self) -> None:
        """Apply Exponential Phase pattern after Layer 3:
        Rz(π/32) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3_all, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_enhanced_cz_pattern(self) -> None:
        """Apply enhanced controlled-Z operations following an exponential distance pattern."""
        # Apply CZ between qubits (i, i+2^k mod n) for i in range(n) and k in [1,2,3]
        for i in range(self.n_qubits):
            for k in [1, 2, 3]:
                distance = 2**k
                if distance < self.n_qubits:  # Ensure the distance is valid
                    target = (i + distance) % self.n_qubits
                    qml.CZ(wires=[i, target])
    
    def _apply_fourier_hadamard_pattern(self) -> None:
        """Apply Fourier-Inspired Hadamard Pattern:
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
            
            # Apply entanglement layers with Exponential Phase pattern
            self._apply_local_entanglement()
            self._apply_exponential_phase1()
            
            self._apply_medium_entanglement()
            self._apply_exponential_phase2()
            
            self._apply_global_exponential_entanglement()
            self._apply_exponential_phase3()
            
            # Apply enhanced controlled-Z pattern
            self._apply_cz_triplets()
            self._apply_enhanced_cz_pattern()
        
        # Apply final encoding layer
        self._encode_features_final_layer(x)
        
        # Apply Fourier-Inspired Hadamard Pattern
        self._apply_fourier_hadamard_pattern()