import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class DiverseGlobalConnectivityFeatureMap(BaseFeatureMap):
    """Diverse Global Connectivity with Progressive Concentration-Aware Phases feature map.

    This feature map combines modified global entanglement with structured phase patterns 
    designed to mitigate exponential concentration effects.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = DiverseGlobalConnectivityFeatureMap(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = 0.92 * np.pi, 
        offset: float = np.pi / 3.5,
        phase1_even: float = np.pi / 6,
        phase1_odd: float = np.pi / 3,
        phase2_even: float = np.pi / 4,
        phase2_odd: float = np.pi / 2,
        phase3_even: float = 2 * np.pi / 3,
        phase3_odd_base: float = 2 * np.pi / 3,
        phase3_odd_step: float = np.pi / 16,
        global_distance_divisor: float = 2.5,
        h_phase_mod0: float = np.pi / 8,
        h_phase_mod1: float = np.pi / 6,
        h_phase_mod2: float = np.pi / 4,
        h_phase_mod3: float = np.pi / 3,
        reps: int = 2
    ) -> None:
        """Initialize the Diverse Global Connectivity feature map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to 0.92*np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/3.5.
            phase1_even (float, optional): Phase shift for even qubits after layer 1. Defaults to np.pi/6.
            phase1_odd (float, optional): Phase shift for odd qubits after layer 1. Defaults to np.pi/3.
            phase2_even (float, optional): Phase shift for even qubits after layer 2. Defaults to np.pi/4.
            phase2_odd (float, optional): Phase shift for odd qubits after layer 2. Defaults to np.pi/2.
            phase3_even (float, optional): Phase shift for even qubits after layer 3. Defaults to 2*np.pi/3.
            phase3_odd_base (float, optional): Base phase for odd qubits after layer 3. Defaults to 2*np.pi/3.
            phase3_odd_step (float, optional): Progressive step for odd phases. Defaults to np.pi/16.
            global_distance_divisor (float, optional): Divisor for global entanglement distance. Defaults to 2.5.
            h_phase_mod0 (float, optional): Phase for mod4=0 qubits in final layer. Defaults to np.pi/8.
            h_phase_mod1 (float, optional): Phase for mod4=1 qubits in final layer. Defaults to np.pi/6.
            h_phase_mod2 (float, optional): Phase for mod4=2 qubits in final layer. Defaults to np.pi/4.
            h_phase_mod3 (float, optional): Phase for mod4=3 qubits in final layer. Defaults to np.pi/3.
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
        self.phase3_even: float = phase3_even
        self.phase3_odd_base: float = phase3_odd_base
        self.phase3_odd_step: float = phase3_odd_step
        self.global_distance_divisor: float = global_distance_divisor
        self.h_phase_mod0: float = h_phase_mod0
        self.h_phase_mod1: float = h_phase_mod1
        self.h_phase_mod2: float = h_phase_mod2
        self.h_phase_mod3: float = h_phase_mod3
        self.reps: int = reps
        
        # Calculate global entanglement distance
        self.global_distance = max(1, int(n_qubits / global_distance_divisor))
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with enhanced Ry distribution.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Ensure we don't go beyond the available features
        n_features = min(80, len(x))
        
        # Features 1-10 → Rx rotations on qubits 1-10
        for i in range(min(10, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 11-30 → Ry rotations on qubits 1-10 (2 layers)
        for i in range(10, min(30, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 10) % self.n_qubits)
        
        # Features 31-40 → Rz rotations on qubits 1-10
        for i in range(30, min(40, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 30) % self.n_qubits)
        
        # Features 41-50 → Rx rotations on qubits 1-10 (second round)
        for i in range(40, min(50, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 40) % self.n_qubits)
        
        # Features 51-80 → Ry rotations on qubits 1-10 (3 more layers)
        for i in range(50, min(80, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 50) % self.n_qubits)

    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_global_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance n/2.5 (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.global_distance) % self.n_qubits])
    
    def _apply_phase1(self) -> None:
        """Apply phase shift after Layer 1: 
        Rz(π/6) to even-indexed qubits and Rz(π/3) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase1_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase1_odd, wires=i)
    
    def _apply_phase2(self) -> None:
        """Apply phase shift after Layer 2: 
        Rz(π/4) to even-indexed qubits and Rz(π/2) to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase2_even, wires=i)
            else:  # odd
                qml.RZ(phi=self.phase2_odd, wires=i)
    
    def _apply_phase3(self) -> None:
        """Apply phase shift after Layer 3: 
        Rz(2π/3) to even-indexed qubits and progressively varied phases to odd-indexed qubits."""
        for i in range(self.n_qubits):
            if i % 2 == 0:  # even
                qml.RZ(phi=self.phase3_even, wires=i)
            else:  # odd with progressive phase
                progressive_phase = self.phase3_odd_base + self.phase3_odd_step * ((i % 4) + 1)
                qml.RZ(phi=progressive_phase, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_modulo4_hadamard(self) -> None:
        """Apply comprehensive modulo-4 Hadamard pattern:
        - Qubit index mod 4 = 0: Apply H gate then Rz(π/8)
        - Qubit index mod 4 = 1: Apply Rz(π/6) then H gate
        - Qubit index mod 4 = 2: Apply H gate then Rz(π/4)
        - Qubit index mod 4 = 3: Apply Rz(π/3) then H gate
        """
        for i in range(self.n_qubits):
            mod4 = i % 4
            if mod4 == 0:
                qml.Hadamard(wires=i)
                qml.RZ(phi=self.h_phase_mod0, wires=i)
            elif mod4 == 1:
                qml.RZ(phi=self.h_phase_mod1, wires=i)
                qml.Hadamard(wires=i)
            elif mod4 == 2:
                qml.Hadamard(wires=i)
                qml.RZ(phi=self.h_phase_mod2, wires=i)
            elif mod4 == 3:
                qml.RZ(phi=self.h_phase_mod3, wires=i)
                qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Hybrid repetition structure
        for _ in range(self.reps):
            # Encode features with Ry-enhanced distribution
            self._encode_features(x)
            
            # Apply entanglement layers with structured phase angles
            self._apply_local_entanglement()
            self._apply_phase1()
            
            self._apply_medium_entanglement()
            self._apply_phase2()
            
            self._apply_global_entanglement()
            self._apply_phase3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Additional feature encoding layer
        self._encode_features(x)
        
        # Apply comprehensive modulo-4 Hadamard pattern
        self._apply_modulo4_hadamard()