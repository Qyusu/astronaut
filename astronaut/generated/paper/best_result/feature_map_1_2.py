import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class InterleavedLayeredEncodingControlledRotationEntanglement(BaseFeatureMap):
    """Interleaved Layered Encoding with Controlled-Rotation Entanglement feature map.
    
    This feature map divides the 80-dimensional input into 4 sequential layers, each handling 20 features.
    In each layer, the first 10 features are used to perform local encoding via RY rotations (with a scaling
    factor s applied to the feature values), and the next 10 features determine the rotation angles for
    entanglement via controlled-RZ (CRZ) gates between neighboring qubits.
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10 and the default scaling parameter s is π.
    """
    
    def __init__(self, n_qubits: int, s: float = np.pi) -> None:
        """Initialize the Interleaved Layered Encoding with Controlled-Rotation Entanglement feature map.
        
        Args:
            n_qubits (int): Number of qubits. For this encoding, n_qubits should ideally be 10.
            s (float): Fixed scaling parameter for rotation angles. Default is π.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        self.s: float = s

    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the feature map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,) corresponding to 4 layers of 20 features each.
        """
        expected_length = 4 * self.n_qubits * 2  # 4 layers * (10 features for RY + 10 for CRZ) = 80 for n_qubits=10
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 4 layers sequentially
        for l in range(4):
            base = 20 * l  # Each layer uses 20 features
            # Local encoding: apply RY rotations for each qubit using the first 10 features of the layer
            for j in range(self.n_qubits):
                qml.RY(phi=self.s * x[base + j], wires=j)
            # Entanglement: apply CRZ gates between each qubit and its neighbor using the next 10 features
            for j in range(self.n_qubits):
                qml.CRZ(phi=self.s * x[base + 10 + j], wires=[j, (j + 1) % self.n_qubits])
