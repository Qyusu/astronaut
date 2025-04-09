import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class PauliStarFeatureMap(BaseFeatureMap):
    """Pauli Feature Map with Star Entanglement.
    
    This feature map systematically encodes 80-dimensional PCA-reduced MNIST data 
    using Pauli rotations (Rx, Ry, Rz) with linearly scaled feature values. The 
    encoding is designed to be independent of qubit count while preserving all feature information.
    
    Args:
        BaseFeatureMap (_type_): base feature map class
    
    Example:
        >>> feature_map = PauliStarFeatureMap(n_qubits=10, scaling_factor=np.pi)
    """

    def __init__(self, n_qubits: int, scaling_factor: float = np.pi) -> None:
        """Initialize the Pauli Star Feature Map class.

        Args:
            n_qubits (int): number of qubits
            scaling_factor (float, optional): Scaling factor for rotation angles. Defaults to np.pi.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scaling_factor: float = scaling_factor
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Divide 80 features into 8 groups of 10 features each
        num_groups = 8
        features_per_group = 10
        
        for g in range(num_groups):
            # Get the features for the current group
            group_start = g * features_per_group
            group_end = group_start + features_per_group
            group_features = x[group_start:group_end]
            
            # Min-max normalization for the group
            group_min = np.min(group_features)
            group_max = np.max(group_features)
            epsilon = 1e-10  # Small epsilon to avoid division by zero
            
            # Apply rotations for this group across all qubits
            for i in range(self.n_qubits):
                # Get feature index for this qubit
                feature_idx = i % len(group_features)
                
                # Scale feature value using min-max normalization
                if np.abs(group_max - group_min) > epsilon:
                    scaled_value = self.scaling_factor * (group_features[feature_idx] - group_min) / (group_max - group_min)
                else:
                    scaled_value = self.scaling_factor * group_features[feature_idx]
                
                # Apply specific rotation pattern based on the group
                if g == 0:  # Group 1: Rx rotations
                    qml.RX(phi=scaled_value, wires=i)
                elif g == 1:  # Group 2: Ry rotations
                    qml.RY(phi=scaled_value, wires=i)
                elif g == 2:  # Group 3: Rz rotations
                    qml.RZ(phi=scaled_value, wires=i)
                elif g == 3:  # Group 4: Rx followed by Ry rotations
                    qml.RX(phi=scaled_value, wires=i)
                    qml.RY(phi=scaled_value, wires=i)
                elif g == 4:  # Group 5: Rx followed by Rz rotations
                    qml.RX(phi=scaled_value, wires=i)
                    qml.RZ(phi=scaled_value, wires=i)
                elif g == 5:  # Group 6: Ry followed by Rz rotations
                    qml.RY(phi=scaled_value, wires=i)
                    qml.RZ(phi=scaled_value, wires=i)
                elif g == 6:  # Group 7: Rx followed by Ry followed by Rz rotations
                    qml.RX(phi=scaled_value, wires=i)
                    qml.RY(phi=scaled_value, wires=i)
                    qml.RZ(phi=scaled_value, wires=i)
                elif g == 7:  # Group 8: Rz followed by Ry followed by Rx rotations
                    qml.RZ(phi=scaled_value, wires=i)
                    qml.RY(phi=scaled_value, wires=i)
                    qml.RX(phi=scaled_value, wires=i)
            
            # Apply 'star' entanglement topology
            # The central qubit rotates with each group
            central_qubit = g % self.n_qubits
            for target_qubit in range(self.n_qubits):
                if target_qubit != central_qubit:
                    qml.CNOT(wires=[central_qubit, target_qubit])