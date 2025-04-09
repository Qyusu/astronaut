import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class FibonacciGoldenFeatureMap(BaseFeatureMap):
    """Custom Final Transformation with Fibonacci-Golden Hybrid Entanglement feature map.
    
    This feature map combines mathematical principles to create a highly expressive yet 
    computationally efficient quantum representation using adaptive Fibonacci-Golden 
    hybrid entanglement pattern and a simplified final transformation.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = FibonacciGoldenFeatureMap(n_qubits=10)
    """

    def __init__(self, n_qubits: int, scale_factor: float = 1.05 * np.pi, offset: float = np.pi/5) -> None:
        """Initialize the Custom Final Transformation with Fibonacci-Golden Hybrid Entanglement feature map.

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
        
        # Calculate golden ratio jump for entanglement
        self.golden_jump = int(self.n_qubits * 0.618)
        
        # Define triplets for first repetition
        # (0,2,5), (1,3,6), (2,4,7), (3,5,8), (4,6,9)
        self.triplets1 = []
        for i in range(5):  # Only need 5 triplets in first repetition
            self.triplets1.append((i, (i+2) % n_qubits, (i+5) % n_qubits))
        
        # Define triplets for second repetition
        # (5,7,0), (6,8,1), (7,9,2), (8,0,3), (9,1,4)
        self.triplets2 = []
        for i in range(5, 10):  # Starting from 5 for 5 triplets
            self.triplets2.append((i, (i+2) % n_qubits, (i+5) % n_qubits))

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

    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

    def _apply_fibonacci2_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (Layer 2, first repetition)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])

    def _apply_fibonacci3_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 3 (Layer 2, second repetition)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 3) % self.n_qubits])

    def _apply_golden_entanglement(self) -> None:
        """Apply CNOT gates following golden ratio pattern (Layer 3, final repetition only)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.golden_jump) % self.n_qubits])

    def _apply_phase1(self) -> None:
        """Apply phase shifts after Layer 1: Rz(π/4) to odd-indexed qubits."""
        for i in range(1, self.n_qubits, 2):  # Odd indices
            qml.RZ(phi=np.pi/4, wires=i)

    def _apply_phase2(self) -> None:
        """Apply phase shifts after Layer 2: Rz(π/6) to even-indexed qubits."""
        for i in range(0, self.n_qubits, 2):  # Even indices
            qml.RZ(phi=np.pi/6, wires=i)

    def _apply_phase3(self) -> None:
        """Apply phase shifts after Layer 3: Rz(π/8) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=np.pi/8, wires=i)

    def _apply_cz_triplets1(self) -> None:
        """Apply controlled-Z gates to triplets for first repetition."""
        for a, b, c in self.triplets1:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])

    def _apply_cz_triplets2(self) -> None:
        """Apply controlled-Z gates to triplets for second repetition."""
        for a, b, c in self.triplets2:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])

    def _apply_simplified_transformation(self) -> None:
        """Apply the simplified final transformation:
        - For even-indexed qubits: Apply Rz(π/4) followed by Hadamard
        - For odd-indexed qubits: Apply Rz(π/3)
        """
        for i in range(self.n_qubits):
            if i % 2 == 0:  # Even indices
                qml.RZ(phi=np.pi/4, wires=i)
                qml.Hadamard(wires=i)
            else:  # Odd indices
                qml.RZ(phi=np.pi/3, wires=i)

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # First repetition
        # Encode features
        self._encode_features(x)
        
        # Apply local entanglement and phase shifts
        self._apply_local_entanglement()
        self._apply_phase1()
        
        # Apply Fibonacci-2 entanglement and phase shifts
        self._apply_fibonacci2_entanglement()
        self._apply_phase2()
        
        # Apply triplets for first repetition
        self._apply_cz_triplets1()
        
        # Second repetition
        # Encode features
        self._encode_features(x)
        
        # Apply local entanglement and phase shifts
        self._apply_local_entanglement()
        self._apply_phase1()
        
        # Apply Fibonacci-3 entanglement and phase shifts
        self._apply_fibonacci3_entanglement()
        self._apply_phase2()
        
        # Apply golden ratio entanglement and phase shifts (only in final repetition)
        self._apply_golden_entanglement()
        self._apply_phase3()
        
        # Apply triplets for second repetition
        self._apply_cz_triplets2()
        
        # Final feature encoding
        self._encode_features(x)
        
        # Apply simplified final transformation
        self._apply_simplified_transformation()