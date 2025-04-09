import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class SeparabilityOptimizedFeatureMap(BaseFeatureMap):
    """Separability-Optimized Adaptive Quantum Map feature map.
    
    This feature map is designed to maximize the separation between different classes
    in the quantum state space while maintaining optimal circuit complexity and interpretability.
    It analyzes feature separability to optimize the encoding and entanglement structure.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = SeparabilityOptimizedFeatureMap(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        repetitions: int = 3, 
        entanglement_density: float = 0.7,
        separability_threshold: float = 0.5
    ) -> None:
        """Initialize the Separability-Optimized Adaptive Quantum Map.

        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions. Defaults to 3.
            entanglement_density (float, optional): Density of entanglement connections (0.0-1.0). Defaults to 0.7.
            separability_threshold (float, optional): Threshold for feature separability score. Defaults to 0.5.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        
        # Hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        self.entanglement_density: float = max(0.0, min(1.0, entanglement_density))
        self.separability_threshold: float = max(0.0, min(1.0, separability_threshold))
        
        # Calculate features per qubit to distribute all 80 features
        # Ensure each qubit gets at least one feature
        self.features_per_qubit: int = max(1, 80 // n_qubits + (1 if 80 % n_qubits > 0 else 0))
        
        # Simulated separability scores and dataset metrics
        self.separability_scores = self._compute_separability_scores()
        self.dataset_separability_index = self._compute_dataset_separability_index()
        
        # Optimized scaling parameters
        self.a_params, self.b_params = self._optimize_scaling_parameters()
        
        # Feature-to-qubit assignments
        self.feature_assignments = self._assign_features_to_qubits()
        
        # Entanglement configuration
        self.entanglement_layers = self._design_entanglement_structure()
        
    def _compute_separability_scores(self) -> np.ndarray:
        """Compute simulated separability scores for the 80 PCA features.
        
        In a real implementation, this would use actual class data to compute
        discriminative power metrics like Fisher's ratio.
        
        Returns:
            np.ndarray: Simulated separability scores for each feature
        """
        # Simulate separability scores: higher for lower-indexed features
        # (assuming PCA components are already sorted by importance)
        scores = np.exp(-0.05 * np.arange(80))
        
        # Add some random variation
        np.random.seed(42)  # For reproducibility
        scores += 0.2 * np.random.rand(80)
        
        # Normalize scores to [0, 1]
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
        return scores
    
    def _compute_dataset_separability_index(self) -> float:
        """Compute simulated dataset separability index.
        
        In a real implementation, this would calculate metrics like
        Separability Index (SI), Hubness Measure Index (HMI), or
        Dataset Separability Index (DSI) from the actual data.
        
        Returns:
            float: Simulated dataset separability index
        """
        # Simulate dataset separability based on the distribution of feature separability scores
        # Higher values indicate easier separation (less entanglement needed)
        separability_index = np.mean(self.separability_scores)
        
        # Adjust based on the number of highly separable features
        high_separability_count = np.sum(self.separability_scores > self.separability_threshold)
        separability_index += 0.2 * (high_separability_count / 80)
        
        return min(1.0, separability_index)
    
    def _optimize_scaling_parameters(self) -> tuple:
        """Optimize linear scaling parameters for each feature.
        
        Returns:
            tuple: (a_params, b_params) for the linear scaling function θ = a*x + b
        """
        a_params = np.zeros(80)
        b_params = np.zeros(80)
        
        # Set scaling parameters based on separability scores
        for i in range(80):
            # For highly discriminative features, use wider range (close to full 2π)
            # For less discriminative features, use narrower range
            a_params[i] = 2 * np.pi * (0.5 + 0.5 * self.separability_scores[i])
            b_params[i] = -a_params[i] / 2  # Center the scaling around 0
        
        return a_params, b_params
    
    def _assign_features_to_qubits(self) -> dict:
        """Assign features to qubits based on separability scores.
        
        Returns:
            dict: Mapping of qubits to lists of assigned features
        """
        # Sort features by separability score (descending)
        sorted_features = np.argsort(-self.separability_scores)
        
        # Assign features to qubits
        assignments = {qubit: [] for qubit in range(self.n_qubits)}
        
        # First pass: assign the top n_qubits features to individual qubits
        for i in range(min(self.n_qubits, 80)):
            assignments[i % self.n_qubits].append(sorted_features[i])
        
        # Second pass: distribute remaining features
        remaining_features = sorted_features[self.n_qubits:] if self.n_qubits < 80 else []
        
        # Strategic assignment of remaining features
        for idx, feature_idx in enumerate(remaining_features):
            # Assign to qubit based on a pattern that distributes features evenly
            target_qubit = idx % self.n_qubits
            if len(assignments[target_qubit]) < self.features_per_qubit:
                assignments[target_qubit].append(feature_idx)
        
        return assignments
    
    def _design_entanglement_structure(self) -> list:
        """Design the entanglement structure based on dataset separability.
        
        Returns:
            list: Entanglement configurations for each layer
        """
        # Adjust entanglement density based on dataset separability
        # Higher separability index → less entanglement needed
        adjusted_density = self.entanglement_density * (1 - 0.5 * self.dataset_separability_index)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define the four entanglement layers
        layers = []
        
        # Layer 1: Local Scale (nearest neighbors)
        layer1 = []
        for i in range(self.n_qubits - 1):
            if np.random.rand() < adjusted_density:
                layer1.append((i, i+1))
        layers.append(layer1)
        
        # Layer 2: Medium Scale (distance-2 connections)
        layer2 = []
        for i in range(self.n_qubits - 2):
            if np.random.rand() < adjusted_density * 0.8:  # Slightly less dense
                layer2.append((i, i+2))
        layers.append(layer2)
        
        # Layer 3: Global Scale (long-range connections)
        layer3 = []
        half_n = self.n_qubits // 2
        for i in range(half_n):
            if np.random.rand() < adjusted_density * 0.6:  # Even less dense
                target = (i + half_n) % self.n_qubits
                layer3.append((i, target))
        layers.append(layer3)
        
        # Layer 4: Higher-Order (three-qubit controlled operations)
        # Implement only if dataset is complex enough (low separability)
        layer4 = []
        if self.dataset_separability_index < 0.7:
            for i in range(self.n_qubits - 2):
                if np.random.rand() < adjusted_density * 0.4:  # Least dense
                    # Control from i, target on i+1, i+2
                    layer4.append((i, i+1, i+2))
        layers.append(layer4)
        
        return layers
    
    def _apply_feature_encoding(self, x: np.ndarray) -> None:
        """Apply encoding rotation gates for all features.
        
        Args:
            x (np.ndarray): input data
        """
        # Apply rotation gates for assigned features on each qubit
        for qubit, features in self.feature_assignments.items():
            for feature_idx in features:
                # Calculate rotation angle using optimized scaling
                angle = self.a_params[feature_idx] * x[feature_idx] + self.b_params[feature_idx]
                
                # Choose rotation gate based on feature separability
                sep_score = self.separability_scores[feature_idx]
                
                # Highly separable features get multiple rotation types
                if sep_score > 0.7:
                    qml.RX(phi=angle, wires=qubit)
                    qml.RZ(phi=angle, wires=qubit)
                elif sep_score > 0.4:
                    qml.RY(phi=angle, wires=qubit)
                else:
                    # Cycle through rotation types for less separable features
                    gate_idx = feature_idx % 3
                    if gate_idx == 0:
                        qml.RX(phi=angle, wires=qubit)
                    elif gate_idx == 1:
                        qml.RY(phi=angle, wires=qubit)
                    else:
                        qml.RZ(phi=angle, wires=qubit)
                        
    def _apply_entanglement_layer(self, layer_idx: int) -> None:
        """Apply entanglement operations for a specific layer.
        
        Args:
            layer_idx (int): Index of the entanglement layer to apply
        """
        # Get entanglement configuration for this layer
        layer = self.entanglement_layers[layer_idx]
        
        if layer_idx < 3:  # Layers 0-2 use CNOT gates
            for control, target in layer:
                qml.CNOT(wires=[control, target])
        else:  # Layer 3 uses controlled-Z on triplets
            for triplet in layer:
                if len(triplet) == 3:
                    # First apply CNOT between control and first target
                    qml.CNOT(wires=[triplet[0], triplet[1]])
                    # Then apply CZ between control and second target
                    qml.CZ(wires=[triplet[0], triplet[2]])
                    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Initialize qubits in superposition
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
        
        # Apply repeated encoding-entanglement blocks
        for rep in range(self.repetitions):
            # Apply feature encoding
            self._apply_feature_encoding(x)
            
            # Apply entanglement layers, except for the final repetition
            if rep < self.repetitions - 1:
                for layer_idx in range(len(self.entanglement_layers)):
                    self._apply_entanglement_layer(layer_idx)
            else:
                # For final repetition, only apply the first two entanglement layers
                # to reduce circuit depth while maintaining expressivity
                for layer_idx in range(min(2, len(self.entanglement_layers))):
                    self._apply_entanglement_layer(layer_idx)