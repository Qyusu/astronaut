import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DualPathNonLocalEntanglementFeatureMap(BaseFeatureMap):
    """Dual-Path Non-Local Entanglement Feature Map.
    
    This feature map segments the 80-dimensional input into 4 layers, each with 20 features.
    In each layer, the first 10 features are used for local encoding via RY rotations on 10 qubits.
    The entanglement stage follows in two parallel pathways:
      - Pathway 1: CRZ gates connect even-indexed qubit pairs (0-1, 2-3, 4-5, 6-7, 8-9) using the next 5 features.
      - Pathway 2: CRY gates connect each even-indexed qubit to the qubit three positions ahead (modulo 10) using the final 5 features.
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Dual-Path Non-Local Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the feature map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 4 layers of 20 features each.
        """
        expected_length = 4 * 20
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        for l in range(4):
            base = 20 * l
            # Local encoding: apply RY rotations using the first 10 features of the layer
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # First entanglement pathway: CRZ gates between even-indexed qubit pairs (0-1, 2-3, etc.)
            for j in range(0, self.n_qubits, 2):
                angle_crz = np.pi * x[base + 10 + (j // 2)]
                qml.CRZ(phi=angle_crz, wires=[j, j + 1])
            
            # Second entanglement pathway: CRY gates connecting even-indexed qubit to the qubit three positions ahead
            for j in range(0, self.n_qubits, 2):
                angle_cry = np.pi * x[base + 15 + (j // 2)]
                qml.CRY(phi=angle_cry, wires=[j, (j + 3) % self.n_qubits])
