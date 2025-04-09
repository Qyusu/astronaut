import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class SimplifiedMultiScaleQuantumEncoder(BaseFeatureMap):
    """Simplified Multi-Scale Quantum Encoder.
    
    A feature map that applies simplified scaling to features and implements
    multi-scale entanglement patterns to capture both local and global relationships.
    
    Args:
        BaseFeatureMap (_type_): base feature map class
        
    Example:
        >>> feature_map = SimplifiedMultiScaleQuantumEncoder(n_qubits=10)
    """
    
    def __init__(self, n_qubits: int, repetitions: int = 3) -> None:
        """Initialize the Simplified Multi-Scale Quantum Encoder.
        
        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions. Defaults to 3.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        
        # Number of features
        self.num_features = 80  # Fixed to 80 for the MNIST PCA features
    
    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply feature encoding with simplified linear scaling.
        
        Args:
            x (np.ndarray): Input feature vector (80,)
        """
        # Distribute all 80 features across qubits in round-robin fashion
        for feature_idx in range(self.num_features):
            # Determine which qubit to apply the rotation to
            qubit_idx = feature_idx % self.n_qubits
            
            # Get feature value
            feature_value = x[feature_idx]
            
            # Apply simplified scaling: θi = π·xi
            angle = np.pi * feature_value
            
            # Determine rotation type based on feature group
            group = feature_idx // self.n_qubits
            
            if group == 0 or group == 3 or group == 6:
                # Features 1-10, 31-40, 61-70 -> Rx
                qml.RX(phi=angle, wires=qubit_idx)
            elif group == 1 or group == 4 or group == 7:
                # Features 11-20, 41-50, 71-80 -> Ry
                qml.RY(phi=angle, wires=qubit_idx)
            else:  # group == 2 or group == 5
                # Features 21-30, 51-60 -> Rz
                qml.RZ(phi=angle, wires=qubit_idx)
    
    def _apply_local_entanglement(self) -> None:
        """Apply local-scale entanglement with CNOT gates between adjacent qubits."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_medium_entanglement(self) -> None:
        """Apply medium-scale entanglement with CNOT gates between qubits separated by distance 2."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_global_entanglement(self) -> None:
        """Apply global-scale entanglement with CNOT gates between distant qubits."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + self.n_qubits // 2) % self.n_qubits])
    
    def _apply_strategic_cz_gates(self) -> None:
        """Apply strategic CZ gates to specific triplets of qubits."""
        # For a 10-qubit system, place CZ gates on specific triplets
        if self.n_qubits == 10:
            triplets = [(0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9)]
        else:
            # For other qubit counts, define appropriate triplets
            # This is a simple approach to ensure triplets cover the entire register
            triplets = []
            step = max(1, self.n_qubits // 10 * 3)  # Scale the step size based on qubit count
            for i in range(0, self.n_qubits, step):
                if i + 2 * step < self.n_qubits:
                    triplets.append((i, (i + step) % self.n_qubits, (i + 2 * step) % self.n_qubits))
        
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
        
        # Repeat the encoding-entanglement block D times
        for _ in range(self.repetitions):
            # Apply feature encoding with simplified scaling
            self._apply_feature_encoding(x)
            
            # Apply multi-scale entanglement
            self._apply_local_entanglement()
            self._apply_medium_entanglement()
            self._apply_global_entanglement()
            
            # Apply strategic CZ gates
            self._apply_strategic_cz_gates()