import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class PhaseOptimizedGoldenRatioEncoder(BaseFeatureMap):
    """Phase-Optimized Golden Ratio Encoder feature map class.
    
    This feature map uses golden ratio-based connectivity and 
    number theory-derived phase shifts to enhance feature interaction.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = PhaseOptimizedGoldenRatioEncoder(n_qubits=10)
    """

    def __init__(self, n_qubits: int, scale_factor: float = 1.05*np.pi, offset: float = np.pi/5) -> None:
        """Initialize the Phase-Optimized Golden Ratio Encoder feature map class.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for rotation angles. Defaults to 1.05*π.
            offset (float, optional): Offset value for rotation angles. Defaults to π/5.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        
        # Calculate golden ratio index jump for global entanglement
        self.golden_jump = int(n_qubits * 0.618)
        
        # Define CZ triplets as specified
        self.cz_triplets = [
            (0,3,6), (1,4,7), (2,5,8), (3,6,9), (4,7,0),
            (5,8,1), (6,9,2), (7,0,3), (8,1,4), (9,2,5)
        ]

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with optimized linear scaling.

        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        rotation_types = [qml.RX, qml.RY, qml.RZ]
        
        # Distribute all features across qubits in a round-robin fashion
        for feature_idx in range(min(80, len(x))):
            qubit_idx = feature_idx % self.n_qubits
            rotation_type_idx = (feature_idx // self.n_qubits) % 3
            rotation_gate = rotation_types[rotation_type_idx]
            
            # Apply optimized linear scaling: 1.05π·xi + π/5
            angle = self.scale_factor * x[feature_idx] + self.offset
            rotation_gate(phi=angle, wires=qubit_idx)
    
    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits in a ring topology (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_phase_shift1(self) -> None:
        """Apply Rz(π/9) to odd-indexed qubits after Layer 1."""
        for i in range(1, self.n_qubits, 2):  # Odd indices
            qml.RZ(phi=np.pi/9, wires=i)
    
    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_phase_shift2(self) -> None:
        """Apply Rz(π/5) to even-indexed qubits after Layer 2."""
        for i in range(0, self.n_qubits, 2):  # Even indices
            qml.RZ(phi=np.pi/5, wires=i)
    
    def _apply_golden_entanglement(self) -> None:
        """Apply CNOT gates following a golden ratio pattern (Layer 3)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.golden_jump) % self.n_qubits])
    
    def _apply_phase_shift3(self) -> None:
        """Apply Rz(3π/7) to all qubits after Layer 3."""
        for i in range(self.n_qubits):
            qml.RZ(phi=3*np.pi/7, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Perform 2 repetitions of encoding-entanglement block
        for _ in range(2):
            # Encode features
            self._encode_features(x)
            
            # Apply three-layer entanglement with phase shifts
            self._apply_local_entanglement()
            self._apply_phase_shift1()
            
            self._apply_medium_entanglement()
            self._apply_phase_shift2()
            
            self._apply_golden_entanglement()
            self._apply_phase_shift3()
            
            # Apply strategic CZ gates
            self._apply_cz_triplets()
        
        # Additional feature encoding layer at the end
        self._encode_features(x)