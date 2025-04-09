import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class FineTunedFourierEncoder(BaseFeatureMap):
    """Fine-Tuned Fourier Encoder with Optimized Scaling feature map.

    This feature map implements a comprehensive Fourier-inspired design with carefully 
    tuned parameters and scaling for rotation gates.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = FineTunedFourierEncoder(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.93 * np.pi, 
        offset: float = np.pi / 3.45,
        phase1_even: float = np.pi / 8,
        phase1_odd: float = np.pi / 4,
        phase2_even: float = np.pi / 4,
        phase2_odd: float = np.pi / 2,
        phase3_all: float = 3 * np.pi / 4,
        h_mod_phase1: float = 3 * np.pi / 16,
        h_mod_phase2: float = 3 * np.pi / 8,
        h_mod_phase3: float = 9 * np.pi / 16,
        reps: int = 2
    ) -> None:
        """Initialize the Fine-Tuned Fourier Encoder with Optimized Scaling feature map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.93*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.45.
            phase1_even (float, optional): Phase for even qubits after layer 1. Defaults to np.pi/8.
            phase1_odd (float, optional): Phase for odd qubits after layer 1. Defaults to np.pi/4.
            phase2_even (float, optional): Phase for even qubits after layer 2. Defaults to np.pi/4.
            phase2_odd (float, optional): Phase for odd qubits after layer 2. Defaults to np.pi/2.
            phase3_all (float, optional): Phase for all qubits after layer 3. Defaults to 3*np.pi/4.
            h_mod_phase1 (float, optional): Phase for mod 4 = 1 qubits before Hadamard. Defaults to 3*np.pi/16.
            h_mod_phase2 (float, optional): Phase for mod 4 = 2 qubits before Hadamard. Defaults to 3*np.pi/8.
            h_mod_phase3 (float, optional): Phase for mod 4 = 3 qubits before Hadamard. Defaults to 9*np.pi/16.
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

    def _encode_features(self, x: np.ndarray, rep: int) -> None:
        """Apply feature encoding for a specific repetition.
        
        Feature distribution:
        * Features 1-10 → Ry rotations on qubits 1-10
        * Features 11-20 → Ry rotations on qubits 1-10 (second layer)
        * Features 21-30 → Ry rotations on qubits 1-10 (third layer)
        * Features 31-40 → Rx rotations on qubits 1-10
        * Features 41-50 → Rx rotations on qubits 1-10 (second layer)
        * Features 51-62 → Ry rotations on qubits 1-10 (partial fourth layer)
        * Features 63-80 → Rz rotations on qubits 1-10 (partial first & second layer)
        
        Args:
            x (np.ndarray): Input data of shape (80,)
            rep (int): Current repetition index
        """
        # Calculate the offset for this repetition (0 for first rep, 40 for second rep)
        offset = rep * 40  # We'll use 40 features per repetition
        
        # Features 1-10 → Ry rotations on qubits 1-10
        for i in range(min(10, len(x) - offset)):
            if offset + i < len(x):
                angle = self.scale_factor * x[offset + i] + self.offset
                qml.RY(phi=angle, wires=i % self.n_qubits)
        
        # Features 11-20 → Ry rotations on qubits 1-10 (second layer)
        for i in range(10, min(20, len(x) - offset)):
            if offset + i < len(x):
                angle = self.scale_factor * x[offset + i] + self.offset
                qml.RY(phi=angle, wires=(i - 10) % self.n_qubits)
        
        # Features 21-30 → Ry rotations on qubits 1-10 (third layer)
        for i in range(20, min(30, len(x) - offset)):
            if offset + i < len(x):
                angle = self.scale_factor * x[offset + i] + self.offset
                qml.RY(phi=angle, wires=(i - 20) % self.n_qubits)
        
        # Features 31-40 → Rx rotations on qubits 1-10
        for i in range(30, min(40, len(x) - offset)):
            if offset + i < len(x):
                angle = self.scale_factor * x[offset + i] + self.offset
                qml.RX(phi=angle, wires=(i - 30) % self.n_qubits)
        
        # If this is the first repetition and we have more than 40 features
        if rep == 0 and len(x) > 40:
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
                wire_idx = (i - 62) % self.n_qubits
                qml.RZ(phi=angle, wires=wire_idx)

    def _encode_final_layer(self, x: np.ndarray) -> None:
        """Apply feature encoding for the final layer.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Use the remaining features for the final layer (reusing if necessary)
        for i in range(self.n_qubits):
            feature_idx = (i + 40) % len(x)  # Reuse features if needed
            angle = self.scale_factor * x[feature_idx] + self.offset
            qml.RY(phi=angle, wires=i)

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
    
    def _apply_fourier_phase1(self) -> None:
        """Apply Fourier Phase pattern after Layer 1:
        Rz(π/8) to even-indexed qubits and Rz(π/4) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_fourier_phase2(self) -> None:
        """Apply Fourier Phase pattern after Layer 2:
        Rz(π/4) to even-indexed qubits and Rz(π/2) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_fourier_phase3(self) -> None:
        """Apply Fourier Phase pattern after Layer 3:
        Rz(3π/4) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3_all, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_optimized_fourier_hadamard(self) -> None:
        """Apply Optimized Fourier Hadamard Pattern:
        - Qubit index mod 4 = 0: Apply H gate
        - Qubit index mod 4 = 1: Apply Rz(3π/16) followed by H gate
        - Qubit index mod 4 = 2: Apply Rz(3π/8) followed by H gate
        - Qubit index mod 4 = 3: Apply Rz(9π/16) followed by H gate
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
            self._encode_features(x, rep)
            
            # Apply entanglement layers with Fourier Phase pattern
            self._apply_local_entanglement()
            self._apply_fourier_phase1()
            
            self._apply_medium_entanglement()
            self._apply_fourier_phase2()
            
            self._apply_global_entanglement()
            self._apply_fourier_phase3()
            
            # Apply controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Apply final encoding layer
        self._encode_final_layer(x)
        
        # Apply Optimized Fourier Hadamard Pattern
        self._apply_optimized_fourier_hadamard()