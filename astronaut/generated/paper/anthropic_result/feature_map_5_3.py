import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class MaximallyConnectedCZNetwork(BaseFeatureMap):
    """Maximally-Connected CZ Network feature map class.
    
    Creates an extensive, optimized network of controlled-Z gates to maximize feature 
    interaction coverage, with specific improvements for noise resilience and effectiveness.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = MaximallyConnectedCZNetwork(n_qubits=10)
    """

    def __init__(self, n_qubits: int, scale_factor: float = 0.75*np.pi, offset: float = np.pi/3) -> None:
        """Initialize the Maximally-Connected CZ Network feature map class.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for rotation angles. Defaults to 0.75*π.
            offset (float, optional): Offset value for rotation angles. Defaults to π/3.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        
        # Generate triplets for CZ network
        self.triplets_set1 = [(i, (i+1) % n_qubits, (i+3) % n_qubits) for i in range(n_qubits)]
        self.triplets_set2 = [(i, (i+2) % n_qubits, (i+5) % n_qubits) for i in range(n_qubits)]

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
            
            # Apply optimized linear scaling: 0.75π·xi + π/3
            angle = self.scale_factor * x[feature_idx] + self.offset
            rotation_gate(phi=angle, wires=qubit_idx)
    
    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits in a ring topology."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_global_entanglement(self) -> None:
        """Apply CNOT gates in a star pattern where qubit 0 controls all other qubits."""
        for i in range(1, self.n_qubits):
            qml.CNOT(wires=[0, i])
    
    def _apply_cz_network(self) -> None:
        """Apply controlled-Z gates based on the optimized triplet selection algorithm."""
        # Apply CZ gates for first set of triplets
        for a, b, c in self.triplets_set1:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])
        
        # Apply CZ gates for second set of triplets
        for a, b, c in self.triplets_set2:
            qml.CZ(wires=[a, b])
            qml.CZ(wires=[b, c])
            qml.CZ(wires=[c, a])
    
    def _apply_final_phase_adjustment(self) -> None:
        """Apply final phase adjustment with Rz(π/2) gates to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=np.pi/2, wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Determine number of repetitions based on feature count
        # For datasets with ≥ 40 features, use 1 repetition, otherwise use 2
        repetitions = 1 if len(x) >= 40 else 2
        
        for _ in range(repetitions):
            # Encode features
            self._encode_features(x)
            
            # Apply two-layer entanglement
            self._apply_local_entanglement()
            self._apply_global_entanglement()
            
            # Apply optimized CZ network
            self._apply_cz_network()
        
        # Apply final phase adjustment
        self._apply_final_phase_adjustment()