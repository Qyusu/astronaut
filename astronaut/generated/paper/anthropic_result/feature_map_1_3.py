import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class MultiScaleFeatureMap(BaseFeatureMap):
    """Multi-scale Feature Map with Entropy-Optimized Linear Embeddings.

    This feature map encodes 80-dimensional data using a multi-scale approach enhanced 
    with information-theoretic insights from recent quantum machine learning research.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = MultiScaleFeatureMap(n_qubits=10, beta=0.5)
    """

    def __init__(self, n_qubits: int, beta: float = 0.5) -> None:
        """Initialize the Multi-scale Feature Map class.

        Args:
            n_qubits (int): number of qubits
            beta (float, optional): Controls scaling variation for global features. Defaults to 0.5.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.beta: float = beta

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Split features into local and global sets
        local_features = x[:40]  # Features 1-40
        global_features = x[40:] # Features 41-80
        
        # ===== LOCAL FEATURES ENCODING =====
        # Group 1: Features 1-10 → Rx rotations
        features_1_10 = local_features[:10]
        for i in range(self.n_qubits):
            feature_idx = i % len(features_1_10)
            # Calculate entropy-informed weighting - we'll use a simple function for ω(i)
            omega = 1 + 0.1 * np.sin(np.pi * i / self.n_qubits)
            # Linear scaling for rotation angle
            max_abs = np.max(np.abs(features_1_10)) + 1e-10  # Avoid division by zero
            angle = (np.pi / (2 * max_abs)) * omega * features_1_10[feature_idx]
            qml.RX(phi=angle, wires=i)
        
        # Group 2: Features 11-20 → Ry rotations
        features_11_20 = local_features[10:20]
        for i in range(self.n_qubits):
            feature_idx = i % len(features_11_20)
            omega = 1 + 0.15 * np.cos(np.pi * i / self.n_qubits)
            max_abs = np.max(np.abs(features_11_20)) + 1e-10
            angle = (np.pi / (2 * max_abs)) * omega * features_11_20[feature_idx]
            qml.RY(phi=angle, wires=i)
        
        # Group 3: Features 21-30 → Balanced compositions of Rx and Ry
        features_21_30 = local_features[20:30]
        for i in range(self.n_qubits):
            feature_idx = i % len(features_21_30)
            omega = 1 + 0.2 * np.sin(2 * np.pi * i / self.n_qubits)
            max_abs = np.max(np.abs(features_21_30)) + 1e-10
            angle = (np.pi / (2 * max_abs)) * omega * features_21_30[feature_idx]
            # Balanced composition of Rx and Ry
            qml.RX(phi=angle/2, wires=i)
            qml.RY(phi=angle/2, wires=i)
        
        # Group 4: Features 31-40 → IQP-inspired encodings
        features_31_40 = local_features[30:40]
        for i in range(self.n_qubits):
            feature_idx = i % len(features_31_40)
            omega = 1 + 0.25 * np.cos(2 * np.pi * i / self.n_qubits)
            max_abs = np.max(np.abs(features_31_40)) + 1e-10
            angle = (np.pi / (2 * max_abs)) * omega * features_31_40[feature_idx]
            # IQP-inspired encoding with Z rotations
            qml.Hadamard(wires=i)
            qml.RZ(phi=angle, wires=i)
            qml.Hadamard(wires=i)
        
        # Apply local entanglement - CNOT gates between neighboring qubits
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, (i+1) % self.n_qubits])
        
        # ===== GLOBAL FEATURES ENCODING =====
        # Group 5: Features 41-50 → Rx rotations with information-preserving scaling
        features_41_50 = global_features[:10]
        for i in range(self.n_qubits):
            feature_idx = i % len(features_41_50)
            scaling_factor = 1 + self.beta * np.sin(np.pi * i / self.n_qubits)
            max_abs = np.max(np.abs(features_41_50)) + 1e-10
            angle = (np.pi / (2 * max_abs)) * scaling_factor * features_41_50[feature_idx]
            qml.RX(phi=angle, wires=i)
        
        # Group 6: Features 51-60 → Ry rotations with information-preserving scaling
        features_51_60 = global_features[10:20]
        for i in range(self.n_qubits):
            feature_idx = i % len(features_51_60)
            scaling_factor = 1 + self.beta * np.cos(np.pi * i / self.n_qubits)
            max_abs = np.max(np.abs(features_51_60)) + 1e-10
            angle = (np.pi / (2 * max_abs)) * scaling_factor * features_51_60[feature_idx]
            qml.RY(phi=angle, wires=i)
        
        # Group 7: Features 61-70 → Structured combinations maintaining moderate entropy correlation
        features_61_70 = global_features[20:30]
        for i in range(self.n_qubits):
            feature_idx = i % len(features_61_70)
            scaling_factor = 1 + self.beta * np.sin(2 * np.pi * i / self.n_qubits)
            max_abs = np.max(np.abs(features_61_70)) + 1e-10
            angle = (np.pi / (2 * max_abs)) * scaling_factor * features_61_70[feature_idx]
            # Structured combination
            qml.RZ(phi=angle/3, wires=i)
            qml.RY(phi=angle/3, wires=i)
            qml.RX(phi=angle/3, wires=i)
        
        # Group 8: Features 71-80 → Complementary encodings maximizing distinctiveness
        features_71_80 = global_features[30:40]
        for i in range(self.n_qubits):
            feature_idx = i % len(features_71_80)
            scaling_factor = 1 + self.beta * np.cos(2 * np.pi * i / self.n_qubits)
            max_abs = np.max(np.abs(features_71_80)) + 1e-10
            angle = (np.pi / (2 * max_abs)) * scaling_factor * features_71_80[feature_idx]
            # Complementary encoding
            qml.Hadamard(wires=i)
            qml.RZ(phi=angle, wires=i)
            qml.RY(phi=angle/2, wires=i)
        
        # Apply global entanglement - CZ gates preserving translational symmetry
        for i in range(self.n_qubits):
            qml.CZ(wires=[i, (i+2) % self.n_qubits])
        
        # Apply balanced entanglement - Additional connections for information retention
        for i in range(0, self.n_qubits, 2):
            qml.CNOT(wires=[i, (i+3) % self.n_qubits])