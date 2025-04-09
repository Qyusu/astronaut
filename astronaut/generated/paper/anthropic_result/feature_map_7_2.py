import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class ChiralityPrimeFeatureMap(BaseFeatureMap):
    """Chirality-Optimized Encoder with Prime Phase Progression.
    
    This feature map introduces direction-dependent entanglement and a systematic 
    phase angle progression based on prime numbers to create a highly expressive 
    quantum representation.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = ChiralityPrimeFeatureMap(n_qubits=10)
    """

    def __init__(self, n_qubits: int, scale_factor: float = 1.05 * np.pi, offset: float = np.pi/5) -> None:
        """Initialize the Chirality-Optimized Encoder with Prime Phase Progression.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for rotation angles. Defaults to 1.05π.
            offset (float, optional): Offset value for rotation angles. Defaults to π/5.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        
        # Calculate golden ratio jump for global entanglement
        self.golden_jump = int(self.n_qubits * 0.618)
        
        # Define prime numbers for phase angles
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        
        # Define CZ triplets for first repetition
        self.cz_triplets1 = []
        for i in range(n_qubits):
            self.cz_triplets1.append((i, (i + 3) % n_qubits, (i + 6) % n_qubits))
        
        # Define CZ triplets for second repetition
        self.cz_triplets2 = []
        for i in range(n_qubits):
            self.cz_triplets2.append((i, (i + 2) % n_qubits, (i + 7) % n_qubits))

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with refined linear scaling.

        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Define rotation gates to use in sequence
        rotation_gates = [qml.RX, qml.RY, qml.RZ]
        
        # Distribute all features across qubits in a round-robin fashion
        for feature_idx in range(min(80, len(x))):
            qubit_idx = feature_idx % self.n_qubits
            rotation_type_idx = (feature_idx // self.n_qubits) % len(rotation_gates)
            rotation_gate = rotation_gates[rotation_type_idx]
            
            # Apply refined linear scaling: 1.05π·xi + π/5
            angle = self.scale_factor * x[feature_idx] + self.offset
            rotation_gate(phi=angle, wires=qubit_idx)
    
    def _apply_local_entanglement_cw(self) -> None:
        """Apply clockwise CNOT gates between adjacent qubits (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_local_entanglement_ccw(self) -> None:
        """Apply counter-clockwise CNOT gates between adjacent qubits (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i - 1) % self.n_qubits])
    
    def _apply_medium_entanglement_cw(self) -> None:
        """Apply clockwise CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_medium_entanglement_ccw(self) -> None:
        """Apply counter-clockwise CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i - 2) % self.n_qubits])
    
    def _apply_golden_entanglement_cw(self) -> None:
        """Apply clockwise CNOT gates following golden ratio pattern (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.golden_jump) % self.n_qubits])
    
    def _apply_golden_entanglement_ccw(self) -> None:
        """Apply counter-clockwise CNOT gates following golden ratio pattern (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i - self.golden_jump) % self.n_qubits])
    
    def _apply_prime_phase_progression(self) -> None:
        """Apply phase shifts with angles derived from prime numbers."""
        for i in range(self.n_qubits):
            prime = self.primes[i % len(self.primes)]
            qml.RZ(phi=np.pi/prime, wires=i)
    
    def _apply_cz_triplets1(self) -> None:
        """Apply controlled-Z gates to the first pattern of triplets."""
        for a, b, c in self.cz_triplets1:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])
    
    def _apply_cz_triplets2(self) -> None:
        """Apply controlled-Z gates to the second pattern of triplets."""
        for a, b, c in self.cz_triplets2:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])
    
    def _apply_final_hadamard(self) -> None:
        """Apply Hadamard gates to all even-indexed qubits."""
        for i in range(0, self.n_qubits, 2):  # Even indices
            qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # First repetition (Clockwise)
        # Encode features
        self._encode_features(x)
        
        # Apply clockwise entanglement with prime phase progression
        self._apply_local_entanglement_cw()
        self._apply_prime_phase_progression()
        
        self._apply_medium_entanglement_cw()
        self._apply_prime_phase_progression()
        
        self._apply_golden_entanglement_cw()
        self._apply_prime_phase_progression()
        
        # Apply first pattern of CZ triplets
        self._apply_cz_triplets1()
        
        # Second repetition (Counter-Clockwise)
        # Encode features
        self._encode_features(x)
        
        # Apply counter-clockwise entanglement with prime phase progression
        self._apply_local_entanglement_ccw()
        self._apply_prime_phase_progression()
        
        self._apply_medium_entanglement_ccw()
        self._apply_prime_phase_progression()
        
        self._apply_golden_entanglement_ccw()
        self._apply_prime_phase_progression()
        
        # Apply second pattern of CZ triplets
        self._apply_cz_triplets2()
        
        # Additional feature encoding layer at the end
        self._encode_features(x)
        
        # Apply final Hadamard layer to even-indexed qubits
        self._apply_final_hadamard()