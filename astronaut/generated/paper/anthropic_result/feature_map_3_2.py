import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class BalancedMultiScaleFeatureMap(BaseFeatureMap):
    """Balanced Multi-Scale Entanglement Map with Concentration Mitigation.

    This feature map implements a balanced approach that captures patterns at multiple scales
    while treating all features equally, with specific mitigations for the exponential 
    concentration problem and noise sensitivity.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = BalancedMultiScaleFeatureMap(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int,
        repetitions: int = None,
        angle_scaling_max: float = np.pi/4
    ) -> None:
        """ "Initialize the Balanced Multi-Scale Entanglement Map.

        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions.
                If None, calculated as min(3, ⌈log₂(m/n)⌉). Defaults to None.
            angle_scaling_max (float, optional): Maximum rotation angle for features.
                Defaults to π/4.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.angle_scaling_max: float = angle_scaling_max
        
        # Calculate repetitions based on formula: D = min(3, ⌈log₂(m/n)⌉)
        self.num_features = 80  # Fixed to 80 for the MNIST PCA features
        if repetitions is None:
            self.repetitions = min(3, int(np.ceil(np.log2(self.num_features / self.n_qubits))))
        else:
            self.repetitions = repetitions
        
        # Statistical parameters (would normally be computed from dataset)
        self.means, self.stds = self._compute_statistics()

    def _compute_statistics(self) -> tuple:
        """Compute simulated mean and standard deviation for each feature.
        
        In a real application, these would be computed from the training data.
        
        Returns:
            tuple: Feature means and standard deviations
        """
        # For simulation purposes, use synthetic statistics
        # Assuming features are normalized to [0, 1], means around 0.5
        means = 0.5 * np.ones(self.num_features)
        
        # Stds in the range [0.1, 0.3]
        np.random.seed(42)  # For reproducibility
        stds = 0.1 + 0.2 * np.random.rand(self.num_features)
        
        return means, stds

    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply feature encoding with angle range regularization.
        
        Args:
            x (np.ndarray): Input feature vector (80,)
        """
        # Distribute all 80 features across qubits in round-robin fashion
        for feature_idx in range(self.num_features):
            # Determine which qubit to apply the rotation to
            qubit_idx = feature_idx % self.n_qubits
            
            # Get feature value and statistics
            feature_value = x[feature_idx]
            mean = self.means[feature_idx]
            std = self.stds[feature_idx]
            
            # Apply angle range regularization:
            # a_i = min(π/(4σ_i), angle_scaling_max)
            # b_i = -a_i * μ_i
            # θ_i = a_i * x_i + b_i
            a_i = min(np.pi / (4 * std), self.angle_scaling_max)
            b_i = -a_i * mean
            angle = a_i * feature_value + b_i
            
            # Determine rotation type based on feature index
            # Use Rx, Ry, Rz in rotation to encode in all bases
            rotation_type = (feature_idx // self.n_qubits) % 3
            
            if rotation_type == 0:
                qml.RX(phi=angle, wires=qubit_idx)
            elif rotation_type == 1:
                qml.RY(phi=angle, wires=qubit_idx)
            else:  # rotation_type == 2
                qml.RZ(phi=angle, wires=qubit_idx)

    def _apply_local_entanglement(self) -> None:
        """Apply local scale entanglement (adjacent qubits)."""
        # Layer 1: Local Scale - Apply CNOT gates between adjacent qubits in a ring topology
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

    def _apply_medium_entanglement(self) -> None:
        """Apply medium scale entanglement (qubits separated by distance 2) with 50% sparsity."""
        # Layer 2: Medium Scale - Apply CNOT gates between qubits separated by distance 2
        # Use 50% sparsity by connecting pairs (i, i+2) where i is even
        for i in range(0, self.n_qubits, 2):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])

    def _apply_global_entanglement(self) -> None:
        """Apply global scale entanglement (distant qubits) with 25% sparsity."""
        # Layer 3: Global Scale - Apply CNOT gates between distant qubits
        # Use 25% sparsity by connecting pairs (i, i+n/2) where i mod 4 = 0
        half_n = self.n_qubits // 2
        for i in range(0, self.n_qubits, 4):
            qml.CNOT(wires=[i, (i + half_n) % self.n_qubits])

    def _apply_triplet_entanglement(self) -> None:
        """Apply controlled-Z gates to selected triplets of qubits."""
        # For each triplet (i, i+1, i+3) where i mod 3 = 0
        for i in range(0, self.n_qubits, 3):
            # Make sure the indices are valid
            j = (i + 1) % self.n_qubits
            k = (i + 3) % self.n_qubits
            
            # Apply CZ gates between all pairs in the triplet
            qml.CZ(wires=[i, j])
            qml.CZ(wires=[j, k])
            qml.CZ(wires=[i, k])

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
            # Apply feature encoding with angle range regularization
            self._apply_feature_encoding(x)
            
            # Apply multi-scale entanglement with structured sparsity
            self._apply_local_entanglement()
            self._apply_medium_entanglement()
            self._apply_global_entanglement()
            self._apply_triplet_entanglement()