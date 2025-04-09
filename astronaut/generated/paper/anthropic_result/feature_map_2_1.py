import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class CorrelationGuidedFeatureMap(BaseFeatureMap):
    """Correlation-Guided Entanglement with Adaptive Scaling feature map.

    This feature map analyzes feature correlations and creates entanglement patterns
    that mirror the statistical structure of the data, with adaptive scaling
    to avoid concentration issues.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = CorrelationGuidedFeatureMap(n_qubits=10, repetitions=2)
    """

    def __init__(
        self, 
        n_qubits: int, 
        repetitions: int = 2, 
        high_corr_threshold: float = 0.7, 
        mod_corr_threshold: float = 0.3
    ) -> None:
        """Initialize the Correlation-Guided feature map.

        Args:
            n_qubits (int): number of qubits
            repetitions (int, optional): Number of encoding-entanglement repetitions. Defaults to 2.
            high_corr_threshold (float, optional): Threshold for high correlation. Defaults to 0.7.
            mod_corr_threshold (float, optional): Threshold for moderate correlation. Defaults to 0.3.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.repetitions: int = repetitions
        self.high_corr_threshold: float = high_corr_threshold
        self.mod_corr_threshold: float = mod_corr_threshold
        
    def _simulate_correlation_analysis(self, x: np.ndarray) -> tuple:
        """Simulates correlation analysis of features.
        
        In a real implementation, this would use dataset statistics. 
        Here we create a deterministic pattern based on feature indices.
        
        Args:
            x (np.ndarray): Input data sample
            
        Returns:
            tuple: Simulated statistics (high_corr_groups, mod_corr_pairs)
        """
        # Create deterministic correlation patterns
        high_corr_groups = []
        mod_corr_pairs = []
        
        # Create high correlation groups based on feature indices
        for i in range(0, 80, 10):
            # Every 10th feature starts a high correlation group
            group_size = min(4, (80 - i))  # Groups of up to 4 features
            high_corr_groups.append(list(range(i, i + group_size)))
        
        # Create moderate correlation pairs based on feature indices
        for i in range(0, 80, 5):
            if i + 2 < 80:  # Ensure the second feature exists
                mod_corr_pairs.append((i, i+2))
        
        return high_corr_groups, mod_corr_pairs

    def _apply_encoding_block(self, x: np.ndarray, rep_index: int) -> None:
        """Apply one block of feature encoding with adaptive scaling.
        
        Args:
            x (np.ndarray): Input data sample
            rep_index (int): Repetition index for variation in encoding
        """
        # Distribute all 80 features across the qubits using round-robin
        features_per_qubit = 80 // self.n_qubits
        
        for qubit in range(self.n_qubits):
            for j in range(features_per_qubit):
                feature_idx = qubit + j * self.n_qubits
                if feature_idx < 80:
                    # For normalized data in [0,1], adapt scaling to spread values across [-π/2, π/2]
                    # Add small phase shift (δ) for concentration avoidance
                    # For repetitions > 0, add variation to angles
                    delta = 0.01 * np.sin(feature_idx + rep_index * 0.1)
                    
                    # Scale from [0,1] to [-π/2, π/2]
                    angle = np.pi * (x[feature_idx] - 0.5) + delta
                    
                    # Use different rotation gates for comprehensive encoding
                    if j % 3 == 0:
                        qml.RX(phi=angle, wires=qubit)
                    elif j % 3 == 1:
                        qml.RY(phi=angle, wires=qubit)
                    else:
                        qml.RZ(phi=angle, wires=qubit)

    def _apply_correlation_entanglement(
        self, 
        high_corr_groups: list, 
        mod_corr_pairs: list
    ) -> None:
        """Apply entanglement based on correlation structure.
        
        Args:
            high_corr_groups (list): Groups of highly correlated features
            mod_corr_pairs (list): Pairs of moderately correlated features
        """
        # Layer 1: Baseline entanglement between neighboring qubits
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        
        # Connect the last qubit with the first to complete the cycle
        qml.CNOT(wires=[self.n_qubits-1, 0])
        
        # Layer 2: Entanglement for moderately correlated feature pairs
        for pair in mod_corr_pairs:
            # Map feature indices to qubit indices (using modulo for wrapping)
            q1 = pair[0] % self.n_qubits
            q2 = pair[1] % self.n_qubits
            if q1 != q2:  # Avoid self-loops
                qml.CNOT(wires=[q1, q2])
        
        # Layer 3: Entanglement for highly correlated feature groups
        for group in high_corr_groups:
            if len(group) >= 3:  # Need at least 3 features for this pattern
                # Map the first three feature indices to qubit indices
                q1 = group[0] % self.n_qubits
                q2 = group[1] % self.n_qubits
                q3 = group[2] % self.n_qubits
                
                # Apply a Toffoli gate (double-controlled X) if all qubits are different
                if q1 != q2 and q2 != q3 and q1 != q3:
                    qml.Toffoli(wires=[q1, q2, q3])
                # Otherwise apply a simpler entangling operation
                elif q1 != q2:
                    qml.CZ(wires=[q1, q2])

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Simulate correlation analysis (in practice would be precomputed from dataset)
        high_corr_groups, mod_corr_pairs = self._simulate_correlation_analysis(x)
        
        # Apply encoding-entanglement blocks multiple times
        for d in range(self.repetitions):
            # Apply feature encoding with adaptive scaling
            self._apply_encoding_block(x, rep_index=d)
            
            # Apply correlation-guided entanglement
            self._apply_correlation_entanglement(high_corr_groups, mod_corr_pairs)