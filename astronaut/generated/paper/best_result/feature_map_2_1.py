import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class MultiScaleAlternatingEntanglementFeatureMap(BaseFeatureMap):
    """Multi-Scale Alternating Entanglement Feature Map.

    This feature map divides the 80-dimensional input into 5 layers, each containing 16 features.
    In each layer l (l = 1,...,5), the first 10 features are encoded onto the 10 qubits via single-qubit RY rotations
    with angles computed as s·π·x. Following the local encoding, a cyclic entanglement stage is applied.
    For each entangling operation, a balanced linear combination of two features is used to calculate the gate angle:
    Let g = 16*(l-1) + 10 + (j mod 6) and g' = 16*(l-1) + 10 + ((j+1) mod 6). The entanglement gate is chosen based
    on the layer: CRZ if l is odd and CRX if l is even.

    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int, s: float = 1.0) -> None:
        """Initialize the Multi-Scale Alternating Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            s (float): Fixed scaling factor for rotation angles. Default is 1.0.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        self.s: float = s

    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the feature map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations using the first 10 features of the layer
            for j in range(self.n_qubits):
                angle = self.s * np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Entanglement stage: apply controlled rotations with a balanced linear combination of features
            for j in range(self.n_qubits):
                index_a = base + 10 + (j % 6)
                index_b = base + 10 + ((j + 1) % 6)
                ent_angle = self.s * np.pi * (0.5 * x[index_a] + 0.5 * x[index_b])
                # Use CRZ if layer (l+1) is odd, CRX if (l+1) is even
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
