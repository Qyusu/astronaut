import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class FibonacciComplementaryTripletsFeatureMap(BaseFeatureMap):
    """Fibonacci-Enhanced Quantum Encoder with Complementary Triplets.
    
    This feature map leverages the mathematical properties of the Fibonacci sequence
    and complementary triplet patterns to create a highly expressive quantum representation.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = FibonacciComplementaryTripletsFeatureMap(n_qubits=10)
    """

    def __init__(self, n_qubits: int, scale_factor: float = np.pi, offset: float = np.pi/4) -> None:
        """Initialize the Fibonacci-Enhanced Quantum Encoder with Complementary Triplets.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for rotation angles. Defaults to π.
            offset (float, optional): Offset value for rotation angles. Defaults to π/4.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        
        # Define canonical triplets for first repetition (CZ gates)
        self.canonical_triplets = []
        for i in range(n_qubits):
            self.canonical_triplets.append((i, (i + 2) % n_qubits, (i + 5) % n_qubits))
        
        # Define complementary triplets for second repetition (CNOT gates)
        self.complementary_triplets = []
        for i in range(n_qubits):
            self.complementary_triplets.append((i, (i + 1) % n_qubits, (i + 4) % n_qubits))

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with optimized linear scaling.

        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Rotation gates to use with emphasis on Ry
        rotation_types = [qml.RY, qml.RX, qml.RZ, qml.RY, qml.RX, qml.RZ, qml.RY, qml.RX]
        
        # Distribute all features across qubits in a round-robin fashion
        for feature_idx in range(min(80, len(x))):
            qubit_idx = feature_idx % self.n_qubits
            rotation_type_idx = (feature_idx // self.n_qubits) % len(rotation_types)
            rotation_gate = rotation_types[rotation_type_idx]
            
            # Apply optimized linear scaling: π·xi + π/4
            angle = self.scale_factor * x[feature_idx] + self.offset
            rotation_gate(phi=angle, wires=qubit_idx)
    
    def _apply_fibonacci_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits in a ring topology (distance 1, first Fibonacci number)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_phase_shift1(self) -> None:
        """Apply Rz(π/9) to odd-indexed qubits after Layer 1."""
        for i in range(1, self.n_qubits, 2):  # Odd indices
            qml.RZ(phi=np.pi/9, wires=i)
    
    def _apply_fibonacci_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (second Fibonacci number)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_phase_shift2(self) -> None:
        """Apply Rz(π/5) to even-indexed qubits after Layer 2."""
        for i in range(0, self.n_qubits, 2):  # Even indices
            qml.RZ(phi=np.pi/5, wires=i)
    
    def _apply_fibonacci_global_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 3 (third Fibonacci number)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 3) % self.n_qubits])
    
    def _apply_phase_shift3(self) -> None:
        """Apply Rz(3π/7) to all qubits after Layer 3."""
        for i in range(self.n_qubits):
            qml.RZ(phi=3*np.pi/7, wires=i)
    
    def _apply_cz_canonical_triplets(self) -> None:
        """Apply controlled-Z gates to canonical triplets (first repetition)."""
        for a, b, c in self.canonical_triplets:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])
    
    def _apply_cnot_complementary_triplets(self) -> None:
        """Apply controlled-NOT gates to complementary triplets (second repetition)."""
        for a, b, c in self.complementary_triplets:
            qml.CNOT(wires=[a, b])
            qml.CNOT(wires=[b, c])
            qml.CNOT(wires=[c, a])
    
    def _apply_final_hadamard(self) -> None:
        """Apply Hadamard gates to all odd-indexed qubits."""
        for i in range(1, self.n_qubits, 2):  # Odd indices
            qml.Hadamard(wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # First repetition
        # Encode features
        self._encode_features(x)
        
        # Apply Fibonacci-based entanglement with phase shifts
        self._apply_fibonacci_local_entanglement()
        self._apply_phase_shift1()
        
        self._apply_fibonacci_medium_entanglement()
        self._apply_phase_shift2()
        
        self._apply_fibonacci_global_entanglement()
        self._apply_phase_shift3()
        
        # Apply canonical triplets with CZ gates
        self._apply_cz_canonical_triplets()
        
        # Second repetition
        # Encode features
        self._encode_features(x)
        
        # Apply Fibonacci-based entanglement with phase shifts
        self._apply_fibonacci_local_entanglement()
        self._apply_phase_shift1()
        
        self._apply_fibonacci_medium_entanglement()
        self._apply_phase_shift2()
        
        self._apply_fibonacci_global_entanglement()
        self._apply_phase_shift3()
        
        # Apply complementary triplets with CNOT gates
        self._apply_cnot_complementary_triplets()
        
        # Additional feature encoding layer at the end
        self._encode_features(x)
        
        # Apply final Hadamard layer to odd-indexed qubits
        self._apply_final_hadamard()