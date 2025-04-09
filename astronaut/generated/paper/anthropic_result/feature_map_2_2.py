import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class PCAImportanceWeightedFeatureMap(BaseFeatureMap):
    """PCA-Importance Weighted Quantum Embeddings with Adaptive Noise Resilience feature map.
    
    This feature map allocates quantum circuit resources based on the relative importance 
    of PCA features as determined by their corresponding eigenvalues, while incorporating 
    strategies to enhance performance on noisy quantum hardware.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = PCAImportanceWeightedFeatureMap(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        tier1_pct: float = 0.5, 
        tier2_pct: float = 0.3,
        tier1_reps: int = 3,
        tier2_reps: int = 2,
        tier3_reps: int = 1
    ) -> None:
        """Initialize the PCA-Importance Weighted feature map.

        Args:
            n_qubits (int): number of qubits
            tier1_pct (float, optional): Percentage of variance for Tier 1 features. Defaults to 0.5.
            tier2_pct (float, optional): Percentage of variance for Tier 2 features. Defaults to 0.3.
            tier1_reps (int, optional): Number of repetitions for Tier 1 features. Defaults to 3.
            tier2_reps (int, optional): Number of repetitions for Tier 2 features. Defaults to 2.
            tier3_reps (int, optional): Number of repetitions for Tier 3 features. Defaults to 1.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.tier1_pct: float = tier1_pct
        self.tier2_pct: float = tier2_pct
        self.tier1_reps: int = tier1_reps
        self.tier2_reps: int = tier2_reps
        self.tier3_reps: int = tier3_reps
        
        # Simulate PCA eigenvalues (decreasing values)
        # In a real implementation, these would come from the PCA analysis
        self.eigenvalues = self._simulate_eigenvalues()
        self.importance_weights = self._calculate_importance_weights()
        self.tier1, self.tier2, self.tier3 = self._assign_feature_tiers()
        
        # Calculate tier assignments for qubits
        self.tier1_qubits = self._get_tier1_qubits()
        self.tier2_qubits = self._get_tier2_qubits()
        self.tier3_qubits = self._get_tier3_qubits()
        
        # Create mappings of features to qubits for each tier
        self.tier1_feature_to_qubit = self._create_feature_to_qubit_mapping(self.tier1, self.tier1_qubits)
        self.tier2_feature_to_qubit = self._create_feature_to_qubit_mapping(self.tier2, self.tier2_qubits)
        self.tier3_feature_to_qubit = self._create_feature_to_qubit_mapping(self.tier3, self.tier3_qubits)
        
    def _simulate_eigenvalues(self) -> np.ndarray:
        """Simulate eigenvalues for PCA components.
        
        In a real implementation, these would come from PCA analysis.
        Here we create a decreasing sequence of values.
        
        Returns:
            np.ndarray: Simulated eigenvalues
        """
        # Create decreasing eigenvalues that follow an exponential decay
        return np.exp(-0.05 * np.arange(80))
    
    def _calculate_importance_weights(self) -> np.ndarray:
        """Calculate normalized importance weights from eigenvalues.
        
        Returns:
            np.ndarray: Normalized importance weights
        """
        # Normalize eigenvalues to sum to 1
        raw_weights = self.eigenvalues / np.sum(self.eigenvalues)
        
        # Ensure a minimum weight for all features
        min_weight = 0.1 * np.mean(raw_weights)
        adjusted_weights = np.maximum(raw_weights, min_weight)
        
        # Re-normalize
        return adjusted_weights / np.sum(adjusted_weights)
    
    def _assign_feature_tiers(self) -> tuple:
        """Assign features to tiers based on cumulative importance.
        
        Returns:
            tuple: Lists of feature indices for each tier
        """
        # Calculate cumulative importance
        cumulative_importance = np.cumsum(self.importance_weights)
        
        # Identify tier boundaries based on cumulative percentages
        tier1_boundary = np.searchsorted(cumulative_importance, self.tier1_pct)
        tier2_boundary = np.searchsorted(cumulative_importance, self.tier1_pct + self.tier2_pct)
        
        # Assign features to tiers
        tier1 = list(range(tier1_boundary))
        tier2 = list(range(tier1_boundary, tier2_boundary))
        tier3 = list(range(tier2_boundary, 80))
        
        return tier1, tier2, tier3
    
    def _get_tier1_qubits(self) -> list:
        """Get qubit indices assigned to Tier 1 features.
        
        Returns:
            list: Qubit indices for Tier 1 features
        """
        # Allocate approximately 1/3 of qubits to Tier 1 (high importance)
        n_tier1_qubits = max(1, self.n_qubits // 3)
        return list(range(n_tier1_qubits))
    
    def _get_tier2_qubits(self) -> list:
        """Get qubit indices assigned to Tier 2 features.
        
        Returns:
            list: Qubit indices for Tier 2 features
        """
        # Allocate approximately 1/3 of qubits to Tier 2 (medium importance)
        n_tier1_qubits = max(1, self.n_qubits // 3)
        n_tier2_qubits = max(1, self.n_qubits // 3)
        return list(range(n_tier1_qubits, n_tier1_qubits + n_tier2_qubits))
    
    def _get_tier3_qubits(self) -> list:
        """Get qubit indices assigned to Tier 3 features.
        
        Returns:
            list: Qubit indices for Tier 3 features
        """
        # Allocate remaining qubits to Tier 3 (low importance)
        n_tier1_qubits = max(1, self.n_qubits // 3)
        n_tier2_qubits = max(1, self.n_qubits // 3)
        return list(range(n_tier1_qubits + n_tier2_qubits, self.n_qubits))
    
    def _create_feature_to_qubit_mapping(self, features: list, qubits: list) -> dict:
        """Create a mapping from feature indices to qubit indices.
        
        Args:
            features (list): List of feature indices
            qubits (list): List of qubit indices
            
        Returns:
            dict: Mapping from qubit indices to lists of feature indices
        """
        mapping = {}
        for i, feature_idx in enumerate(features):
            qubit_idx = qubits[i % len(qubits)]  # Distribute features evenly among available qubits
            if qubit_idx not in mapping:
                mapping[qubit_idx] = []
            mapping[qubit_idx].append(feature_idx)
        return mapping
    
    def _apply_tier1_encoding(self, x: np.ndarray) -> None:
        """Apply encoding for Tier 1 (high importance) features.
        
        Args:
            x (np.ndarray): Input data
        """
        # Apply all three rotation gates (Rx, Ry, Rz) to each feature
        for qubit, features in self.tier1_feature_to_qubit.items():
            for feature_idx in features:
                # Scale by importance weight
                weight = self.importance_weights[feature_idx]
                angle = np.pi * weight * x[feature_idx] - np.pi/2
                
                # Apply all three rotations for Tier 1 features
                qml.RX(phi=angle, wires=qubit)
                qml.RY(phi=angle, wires=qubit)
                qml.RZ(phi=angle, wires=qubit)
        
        # Add ZZ-interaction inspired encoding between tier 1 qubits
        if len(self.tier1_qubits) >= 2:
            for i in range(len(self.tier1_qubits) - 1):
                q1 = self.tier1_qubits[i]
                q2 = self.tier1_qubits[i + 1]
                # Use the average importance weight of features on these qubits
                avg_weight = 0.0
                count = 0
                for feature_idx in self.tier1_feature_to_qubit.get(q1, []):
                    avg_weight += self.importance_weights[feature_idx]
                    count += 1
                for feature_idx in self.tier1_feature_to_qubit.get(q2, []):
                    avg_weight += self.importance_weights[feature_idx]
                    count += 1
                
                if count > 0:
                    avg_weight /= count
                    zz_angle = np.pi/2 * avg_weight
                    qml.IsingZZ(phi=zz_angle, wires=[q1, q2])
    
    def _apply_tier2_encoding(self, x: np.ndarray) -> None:
        """Apply encoding for Tier 2 (medium importance) features.
        
        Args:
            x (np.ndarray): Input data
        """
        # Apply two rotation gates to each feature
        for qubit, features in self.tier2_feature_to_qubit.items():
            for i, feature_idx in enumerate(features):
                # Scale by importance weight
                weight = self.importance_weights[feature_idx]
                angle = np.pi * weight * x[feature_idx] - np.pi/2
                
                # Select two out of three rotations based on feature index
                if i % 3 == 0:
                    qml.RX(phi=angle, wires=qubit)
                    qml.RY(phi=angle, wires=qubit)
                elif i % 3 == 1:
                    qml.RX(phi=angle, wires=qubit)
                    qml.RZ(phi=angle, wires=qubit)
                else:
                    qml.RY(phi=angle, wires=qubit)
                    qml.RZ(phi=angle, wires=qubit)
    
    def _apply_tier3_encoding(self, x: np.ndarray) -> None:
        """Apply encoding for Tier 3 (low importance) features.
        
        Args:
            x (np.ndarray): Input data
        """
        # Apply single rotation gate to each feature
        for qubit, features in self.tier3_feature_to_qubit.items():
            for i, feature_idx in enumerate(features):
                # Scale by importance weight
                weight = self.importance_weights[feature_idx]
                angle = np.pi * weight * x[feature_idx] - np.pi/2
                
                # Select one rotation based on feature index
                if i % 3 == 0:
                    qml.RX(phi=angle, wires=qubit)
                elif i % 3 == 1:
                    qml.RY(phi=angle, wires=qubit)
                else:
                    qml.RZ(phi=angle, wires=qubit)
    
    def _apply_hierarchical_entanglement(self) -> None:
        """Apply correlation-aware hierarchical entanglement between qubits."""
        # 1. Apply nearest-neighbor CNOT gates within tiers
        self._apply_nearest_neighbor_entanglement(self.tier1_qubits)
        self._apply_nearest_neighbor_entanglement(self.tier2_qubits)
        self._apply_nearest_neighbor_entanglement(self.tier3_qubits)
        
        # 2. Apply long-range entanglement between tiers
        self._apply_cross_tier_entanglement(self.tier1_qubits, self.tier2_qubits)
        self._apply_cross_tier_entanglement(self.tier1_qubits, self.tier3_qubits)
        self._apply_cross_tier_entanglement(self.tier2_qubits, self.tier3_qubits)
    
    def _apply_nearest_neighbor_entanglement(self, qubit_list: list) -> None:
        """Apply nearest-neighbor CNOT gates within a tier.
        
        Args:
            qubit_list (list): List of qubit indices in the tier
        """
        if len(qubit_list) < 2:
            return  # Need at least 2 qubits for entanglement
            
        # Apply CNOT between neighboring qubits
        for i in range(len(qubit_list) - 1):
            qml.CNOT(wires=[qubit_list[i], qubit_list[i+1]])
        
        # Connect last and first qubits to complete the cycle (if more than 2 qubits)
        if len(qubit_list) > 2:
            qml.CNOT(wires=[qubit_list[-1], qubit_list[0]])
    
    def _apply_cross_tier_entanglement(self, control_qubits: list, target_qubits: list) -> None:
        """Apply entanglement between qubits in different tiers.
        
        Args:
            control_qubits (list): List of qubit indices for control operations
            target_qubits (list): List of qubit indices for target operations
        """
        if not control_qubits or not target_qubits:
            return  # Need qubits in both tiers
            
        # Apply controlled operations between tiers
        # Higher tier (control) to lower tier (target)
        min_pairs = min(len(control_qubits), len(target_qubits))
        for i in range(min_pairs):
            # Apply CNOT with control from higher tier to target in lower tier
            qml.CNOT(wires=[control_qubits[i % len(control_qubits)], 
                          target_qubits[i % len(target_qubits)]])
            
            # For additional entanglement (less frequent)
            if i % 2 == 0 and i < min_pairs - 1:
                # Apply controlled rotation with control from higher tier
                control = control_qubits[i % len(control_qubits)]
                target = target_qubits[(i+1) % len(target_qubits)]
                qml.CRZ(phi=np.pi/2, wires=[control, target])
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Apply weighted encoding and entanglement with tier-specific repetitions
        
        # Tier 1 encoding and entanglement (highest priority features)
        for _ in range(self.tier1_reps):
            self._apply_tier1_encoding(x)
            self._apply_hierarchical_entanglement()
        
        # Tier 2 encoding and entanglement (medium priority features)
        for _ in range(self.tier2_reps):
            self._apply_tier2_encoding(x)
            self._apply_hierarchical_entanglement()
        
        # Tier 3 encoding and entanglement (lowest priority features)
        for _ in range(self.tier3_reps):
            self._apply_tier3_encoding(x)
            self._apply_hierarchical_entanglement()