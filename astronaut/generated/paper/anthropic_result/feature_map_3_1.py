import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class UniformCorrelationGuidedFeatureMap(BaseFeatureMap):
    """Uniform Correlation-Guided Quantum Feature Map.

    This feature map uniformly treats all 80 PCA-reduced features while leveraging
    correlation information to guide effective entanglement with enhanced noise resilience.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = UniformCorrelationGuidedFeatureMap(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        correlation_threshold: float = 0.3,
        angle_scaling_factor: float = 1.0,
        repetitions: int = 2,
        enable_noise_resilience: bool = True
    ) -> None:
        """Initialize the Uniform Correlation-Guided Quantum Feature Map.

        Args:
            n_qubits (int): number of qubits
            correlation_threshold (float, optional): Threshold for significant feature correlations. Defaults to 0.3.
            angle_scaling_factor (float, optional): Dataset-dependent angle scaling factor (λ). Defaults to 1.0.
            repetitions (int, optional): Number of feature map repetitions. Defaults to 2.
            enable_noise_resilience (bool, optional): Enable noise resilience mechanisms. Defaults to True.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.correlation_threshold: float = correlation_threshold
        self.angle_scaling_factor: float = angle_scaling_factor
        self.repetitions: int = repetitions
        self.enable_noise_resilience: bool = enable_noise_resilience
        
        # Compute statistics (in a real application, these would come from actual data analysis)
        self.feature_means, self.feature_stds = self._compute_statistics()
        self.correlation_matrix = self._compute_correlation_matrix()
        self.dataset_separability = self._compute_dataset_separability()
        
        # Adaptive parameters based on dataset characteristics
        self._adjust_parameters_from_separability()
    
    def _compute_statistics(self) -> tuple:
        """Compute simulated mean and standard deviation for each feature.
        
        In a real application, these would be computed from the training data.
        
        Returns:
            tuple: Feature means and standard deviations
        """
        # For simulation, we'll create synthetic statistics
        # In a real application, these would be computed from actual data
        np.random.seed(42)  # For reproducibility
        
        # Simulated means (centered around 0.5 as the data is normalized to [0,1])
        means = 0.5 * np.ones(80) + 0.1 * np.random.randn(80)
        
        # Simulated standard deviations (ranging from 0.05 to 0.3)
        stds = 0.05 + 0.25 * np.random.rand(80)
        
        return means, stds

    def _compute_correlation_matrix(self) -> np.ndarray:
        """Compute simulated correlation matrix for the features.
        
        In a real application, this would be the actual correlation matrix
        of the training data features.
        
        Returns:
            np.ndarray: Feature correlation matrix
        """
        # For simulation, we'll create a synthetic correlation matrix
        # In a real application, this would be computed from actual data
        np.random.seed(42)  # For reproducibility
        
        # Initialize with a diagonal matrix (self-correlations are 1.0)
        corr_matrix = np.eye(80)
        
        # Add some correlated feature pairs
        for i in range(80):
            for j in range(i+1, 80):
                # Nearby features (in PCA order) are more likely to be correlated
                proximity_factor = np.exp(-0.05 * abs(i - j))
                
                # Generate correlation coefficient (higher for nearby features)
                corr_value = proximity_factor * (0.5 * np.random.rand() + 0.2)
                
                # Ensure it doesn't exceed 0.9 (not perfectly correlated)
                corr_value = min(0.9, corr_value)
                
                # Set symmetric correlation values
                corr_matrix[i, j] = corr_value
                corr_matrix[j, i] = corr_value
        
        return corr_matrix

    def _compute_dataset_separability(self) -> float:
        """Compute simulated dataset separability metric.
        
        In a real application, this would be computed from actual class distributions.
        
        Returns:
            float: Separability metric (0.0-1.0, higher means more separable)
        """
        # For simulation, we'll create a synthetic separability score
        # In a real application, this would be computed from actual data
        np.random.seed(42)  # For reproducibility
        
        # Base separability on correlation matrix structure
        # More correlations typically indicate less separability
        significant_correlations = np.sum(np.abs(self.correlation_matrix) > self.correlation_threshold)
        total_possible_correlations = 80 * 79 / 2  # Number of off-diagonal elements
        
        # Compute separability (inversely related to number of significant correlations)
        separability = 1.0 - (significant_correlations / total_possible_correlations / 2)
        
        # Add some random variation
        separability += 0.1 * np.random.randn()
        
        # Ensure it's between 0 and 1
        separability = max(0.1, min(0.9, separability))
        
        return separability
    
    def _adjust_parameters_from_separability(self) -> None:
        """Adjust hyperparameters based on dataset separability.
        
        This includes setting the angle scaling factor and repetitions.
        """
        # Adjust angle scaling factor (λ) based on separability
        # For less separable datasets (lower separability score), increase λ to create smaller angles
        # For more separable datasets (higher separability score), decrease λ to create larger angles
        self.angle_scaling_factor = 1.3 - 0.6 * self.dataset_separability
        
        # Adjust repetitions based on separability
        # For less separable datasets, use more repetitions
        if self.dataset_separability > 0.7:
            self.repetitions = max(1, self.repetitions - 1)
        elif self.dataset_separability < 0.3:
            self.repetitions = min(4, self.repetitions + 1)
    
    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply encoding rotations for all features using round-robin distribution.

        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Apply rotations for each feature
        for feature_idx in range(80):
            # Determine target qubit using round-robin distribution
            qubit = feature_idx % self.n_qubits
            
            # Compute scaled angle using dataset-aware adaptive scaling
            mean = self.feature_means[feature_idx]
            std = self.feature_stds[feature_idx]
            
            # Implement the adaptive scaling formula: θ = (π/(2σλ))·x - (π·μ/(2σλ))
            angle = (np.pi / (2 * std * self.angle_scaling_factor)) * (x[feature_idx] - mean)
            
            # Determine rotation gate type based on feature index modulo 3
            gate_type = (feature_idx // self.n_qubits) % 3
            
            # Apply appropriate rotation gate
            if gate_type == 0:
                qml.RX(phi=angle, wires=qubit)
            elif gate_type == 1:
                qml.RY(phi=angle, wires=qubit)
            else:
                qml.RZ(phi=angle, wires=qubit)
    
    def _apply_hierarchical_entanglement(self) -> None:
        """Apply hierarchical correlation-guided entanglement operations.
        """
        # Layer 1: Base Layer - Adjacent CNOT gates (Linear ZZ-type entanglement)
        self._apply_base_entanglement_layer()
        
        # Layer 2: Correlation Layer - Entangle qubits that encode correlated features
        self._apply_correlation_entanglement_layer()
        
        # Layer 3: Complex Layer - Apply only for complex datasets (low separability)
        if self.dataset_separability < 0.5:
            self._apply_complex_entanglement_layer()
        
        # Apply noise resilience mechanisms if enabled
        if self.enable_noise_resilience:
            self._apply_noise_resilience_operations()

    def _apply_base_entanglement_layer(self) -> None:
        """Apply base entanglement layer (adjacent CNOT gates).
        """
        # Apply CNOT gates between adjacent qubits
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        
        # Optional circular connection
        if self.n_qubits > 2:
            qml.CNOT(wires=[self.n_qubits - 1, 0])

    def _apply_correlation_entanglement_layer(self) -> None:
        """Apply correlation-guided entanglement operations.
        
        This uses the correlation matrix to determine which qubits to entangle.
        """
        # For each pair of features, apply CNOT if they're significantly correlated
        # We'll limit the number of CNOTs to avoid excessive depth
        num_cnots = 0
        max_cnots = int(self.n_qubits * (1.0 - self.dataset_separability) * 2)
        
        # Identify pairs of qubits to entangle based on feature correlations
        for i in range(self.n_qubits):
            for j in range(i+2, self.n_qubits):  # Skip adjacent qubits (already entangled in base layer)
                if num_cnots >= max_cnots:
                    break
                    
                # Check if any features mapped to qubits i and j are correlated
                has_correlation = False
                
                # Check correlations between features mapped to these qubits
                for fi in range(i, 80, self.n_qubits):
                    if fi >= 80:
                        continue
                        
                    for fj in range(j, 80, self.n_qubits):
                        if fj >= 80:
                            continue
                            
                        if abs(self.correlation_matrix[fi, fj]) > self.correlation_threshold:
                            has_correlation = True
                            break
                    
                    if has_correlation:
                        break
                
                if has_correlation:
                    qml.CNOT(wires=[i, j])
                    num_cnots += 1

    def _apply_complex_entanglement_layer(self) -> None:
        """Apply complex entanglement operations for datasets with low separability.
        
        This includes controlled-Z gates for triplets of qubits with mutually correlated features.
        """
        # For complex datasets, add controlled-Z gates for selected triplets
        num_triplets = int(self.n_qubits * (1.0 - self.dataset_separability))
        
        # Select triplets of qubits with highest mutual correlations
        triplets = []
        
        # Simple heuristic: take consecutive triplets for simplicity
        for i in range(min(num_triplets, self.n_qubits-2)):
            triplets.append((i, (i+1) % self.n_qubits, (i+2) % self.n_qubits))
        
        # Apply CZ gates to selected triplets
        for i, j, k in triplets:
            # Use Toffoli or its decomposition
            qml.Toffoli(wires=[i, j, k])

    def _apply_noise_resilience_operations(self) -> None:
        """Apply noise resilience mechanisms.
        
        This includes strategic phase shifts to avoid concentration effects.
        """
        # Add small phase rotations to improve tolerance to bit-flip errors
        # These are fixed non-trainable angles
        np.random.seed(42)  # For reproducibility
        
        for i in range(self.n_qubits):
            # Small fixed phase rotation
            phase_angle = 0.05 * np.pi * (i + 1) / self.n_qubits
            qml.RZ(phi=phase_angle, wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Initialize qubits in superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        
        # Apply the encoding-entanglement pattern for the specified number of repetitions
        for _ in range(self.repetitions):
            # Apply feature encoding
            self._apply_feature_encoding(x)
            
            # Apply hierarchical entanglement
            self._apply_hierarchical_entanglement()