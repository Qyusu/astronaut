import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class EnhancedStatisticalQuantumKernel(BaseFeatureMap):
    """Enhanced Statistical Quantum Kernel with Variance Safeguard.
    
    This feature map incorporates statistical properties of the dataset to adaptively
    scale features and create targeted entanglement based on feature correlations,
    with a safeguard mechanism for low-variance features.
    
    Args:
        BaseFeatureMap (_type_): base feature map class
        
    Example:
        >>> feature_map = EnhancedStatisticalQuantumKernel(n_qubits=10)
    """
    
    def __init__(
        self, 
        n_qubits: int, 
        repetitions: int = 2,
        corr_threshold: float = 0.5,
        min_variance_factor: float = 0.01
    ) -> None:
        """Initialize the Enhanced Statistical Quantum Kernel with Variance Safeguard.
        
        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions. Defaults to 2.
            corr_threshold (float, optional): Correlation threshold for targeted entanglement. Defaults to 0.5.
            min_variance_factor (float, optional): Factor multiplied by max std to get min variance threshold. Defaults to 0.01.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        
        # Hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        self.corr_threshold: float = corr_threshold
        self.min_variance_factor: float = min_variance_factor
        
        # Number of features
        self.num_features = 80  # Fixed to 80 for the MNIST PCA features
        
        # Calculate top K pairs (typically set to n_qubits)
        self.top_k = n_qubits
        
        # Compute statistical parameters for feature scaling
        self.means, self.stds, self.correlated_pairs = self._compute_statistics()
        
    def _compute_statistics(self) -> tuple:
        """Compute simulated mean, standard deviation, and correlation for each feature.
        
        In a real application, these would be computed from the training data.
        
        Returns:
            tuple: Feature means, standard deviations, and top correlated pairs
        """
        # For simulation purposes, use synthetic statistics
        # Assuming features are normalized to [0, 1], means around 0.5
        means = 0.5 * np.ones(self.num_features)
        
        # Stds in the range [0.1, 0.3]
        np.random.seed(42)  # For reproducibility
        stds = 0.1 + 0.2 * np.random.rand(self.num_features)
        
        # Generate synthetic correlation matrix (in real scenario, this would be computed from data)
        # Just for simulation - create a random correlation matrix
        np.random.seed(42)  # For reproducibility
        corr_matrix = np.eye(self.num_features)  # Start with identity matrix
        
        # Add some correlations
        for i in range(self.num_features):
            for j in range(i+1, self.num_features):
                if np.random.rand() > 0.8:  # 20% chance of strong correlation
                    r = 0.5 + 0.5 * np.random.rand()  # Correlation between 0.5 and 1.0
                else:
                    r = 0.2 * np.random.rand()  # Weak correlation between 0 and 0.2
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
        
        # Find top-K most correlated pairs
        correlated_pairs = []
        for i in range(self.num_features):
            for j in range(i+1, self.num_features):
                if corr_matrix[i, j] > self.corr_threshold:
                    correlated_pairs.append((i, j, corr_matrix[i, j]))
        
        # Sort by correlation value (highest first)
        correlated_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Take only top K pairs
        correlated_pairs = correlated_pairs[:self.top_k]
        
        return means, stds, correlated_pairs
        
    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply feature encoding with adaptive statistical scaling and variance safeguard.
        
        Args:
            x (np.ndarray): Input feature vector (80,)
        """
        # Calculate minimum variance threshold
        epsilon = self.min_variance_factor * np.max(self.stds)
        
        # Distribute all 80 features across qubits in round-robin fashion
        for feature_idx in range(self.num_features):
            # Determine which qubit to apply the rotation to
            qubit_idx = feature_idx % self.n_qubits
            
            # Get feature value and statistics
            feature_value = x[feature_idx]
            mean = self.means[feature_idx]
            std = self.stds[feature_idx]
            
            # Apply adaptive statistical scaling with variance safeguard
            if std >= epsilon:
                a_i = np.pi / (2 * std)
                b_i = -np.pi * mean / (2 * std)
            else:
                # For low-variance features, use minimum variance threshold
                a_i = np.pi / (2 * epsilon)
                b_i = -np.pi * mean / (2 * epsilon)
                
            angle = a_i * feature_value + b_i
            
            # Determine rotation pattern based on feature index
            pattern_idx = feature_idx // self.n_qubits
            
            if pattern_idx == 0:
                # Features 1-10: Rx
                qml.RX(phi=angle, wires=qubit_idx)
            elif pattern_idx == 1:
                # Features 11-20: Ry
                qml.RY(phi=angle, wires=qubit_idx)
            elif pattern_idx == 2:
                # Features 21-30: Rz
                qml.RZ(phi=angle, wires=qubit_idx)
            elif pattern_idx == 3:
                # Features 31-40: Rx then Ry
                qml.RX(phi=angle, wires=qubit_idx)
                qml.RY(phi=angle/2, wires=qubit_idx)  # Using angle/2 for the second rotation to reduce total rotation
            elif pattern_idx == 4:
                # Features 41-50: Rx then Rz
                qml.RX(phi=angle, wires=qubit_idx)
                qml.RZ(phi=angle/2, wires=qubit_idx)
            elif pattern_idx == 5:
                # Features 51-60: Ry then Rz
                qml.RY(phi=angle, wires=qubit_idx)
                qml.RZ(phi=angle/2, wires=qubit_idx)
            elif pattern_idx == 6:
                # Features 61-70: Rx then Ry then Rz
                qml.RX(phi=angle, wires=qubit_idx)
                qml.RY(phi=angle/2, wires=qubit_idx)
                qml.RZ(phi=angle/3, wires=qubit_idx)
            else:  # pattern_idx == 7
                # Features 71-80: Rz then Ry then Rx
                qml.RZ(phi=angle, wires=qubit_idx)
                qml.RY(phi=angle/2, wires=qubit_idx)
                qml.RX(phi=angle/3, wires=qubit_idx)
    
    def _apply_nearest_neighbor_entanglement(self) -> None:
        """Apply nearest-neighbor entanglement with CNOT gates in a ring topology."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_targeted_entanglement(self) -> None:
        """Apply targeted entanglement using CZ gates for correlated feature pairs."""
        # Map feature index to qubit index
        for feature_i, feature_j, _ in self.correlated_pairs:
            qubit_i = feature_i % self.n_qubits
            qubit_j = feature_j % self.n_qubits
            
            # Only apply if mapped to different qubits
            if qubit_i != qubit_j:
                qml.CZ(wires=[qubit_i, qubit_j])
    
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
            # Apply feature encoding with adaptive statistical scaling
            self._apply_feature_encoding(x)
            
            # Apply nearest-neighbor entanglement
            self._apply_nearest_neighbor_entanglement()
            
            # Apply targeted entanglement for correlated features
            self._apply_targeted_entanglement()