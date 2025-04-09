import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class BalancedThreeLayerQuantumEncoder(BaseFeatureMap):
    """Balanced Three-Layer Quantum Encoder.
    
    A feature map that applies a structured three-layer entanglement pattern
    to capture correlations at different scales.
    
    Args:
        BaseFeatureMap (_type_): base feature map class
        
    Example:
        >>> feature_map = BalancedThreeLayerQuantumEncoder(n_qubits=10)
    """
    
    def __init__(self, n_qubits: int, repetitions: int = 2) -> None:
        """Initialize the Balanced Three-Layer Quantum Encoder.
        
        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions. Defaults to 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        
        # Number of features
        self.num_features = 80  # Fixed to 80 for the MNIST PCA features
    
    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply feature encoding with linear scaling and offset.
        
        Args:
            x (np.ndarray): Input feature vector (80,)
        """
        # Distribute all 80 features across qubits in round-robin fashion
        for feature_idx in range(self.num_features):
            # Determine which qubit to apply the rotation to
            qubit_idx = feature_idx % self.n_qubits
            
            # Get feature value
            feature_value = x[feature_idx]
            
            # Apply scaling with offset: θi = π·xi + π/4
            angle = np.pi * feature_value + np.pi / 4
            
            # Determine rotation type based on feature group
            group = (feature_idx // self.n_qubits) % 3
            
            if group == 0:
                # Use Rx for first group of features
                qml.RX(phi=angle, wires=qubit_idx)
            elif group == 1:
                # Use Ry for second group of features
                qml.RY(phi=angle, wires=qubit_idx)
            else:  # group == 2
                # Use Rz for third group of features
                qml.RZ(phi=angle, wires=qubit_idx)
    
    def _apply_local_entanglement(self) -> None:
        """Apply local entanglement with CNOT gates between pairs of adjacent qubits."""
        for i in range(0, self.n_qubits, 2):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_medium_entanglement(self) -> None:
        """Apply medium entanglement with CNOT gates between qubits separated by distance 2."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_global_entanglement(self) -> None:
        """Apply global entanglement with CNOT gates between distant qubits."""
        for i in range(self.n_qubits // 2):
            qml.CNOT(wires=[i, (i + self.n_qubits // 2) % self.n_qubits])
    
    def _apply_strategic_cz_gates(self) -> None:
        """Apply strategic CZ gates to selected triplets of qubits."""
        # For a 10-qubit system, define the specific triplets
        if self.n_qubits == 10:
            triplets = [(0, 2, 5), (1, 3, 6), (2, 4, 7), (3, 5, 8), (4, 6, 9)]
        else:
            # For other qubit counts, define appropriate triplets
            triplets = []
            for i in range(self.n_qubits):
                # Create triplets that include local, medium, and global connections
                triplets.append((i, (i + 2) % self.n_qubits, (i + self.n_qubits // 2) % self.n_qubits))
        
        # Apply CZ gates to each triplet
        for triplet in triplets:
            # Apply CZ between first and second qubit
            qml.CZ(wires=[triplet[0], triplet[1]])
            # Apply CZ between second and third qubit
            qml.CZ(wires=[triplet[1], triplet[2]])
            # Apply CZ between first and third qubit
            qml.CZ(wires=[triplet[0], triplet[2]])
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Initialize all qubits in superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        
        # Repeat the encoding-entanglement block the specified number of times
        for _ in range(self.repetitions):
            # Apply feature encoding with linear scaling and offset
            self._apply_feature_encoding(x)
            
            # Apply the three-layer entanglement structure
            self._apply_local_entanglement()
            self._apply_medium_entanglement()
            self._apply_global_entanglement()
            
            # Apply strategic CZ gates
            self._apply_strategic_cz_gates()