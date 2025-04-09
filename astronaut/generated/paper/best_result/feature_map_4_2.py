import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class FourLayerStandardLocalEntanglementFeatureMap(BaseFeatureMap):
    """4-Layer Standard Local Entanglement Feature Map.
    
    This feature map divides the 80-dimensional input into 4 layers, each containing 20 features.
    For each layer l (l = 1,...,4):
      - The first 10 features are encoded on 10 qubits via single-qubit RY rotations with angles computed as π * x,
        where for qubit j the rotation is RY(π * x[20*(l-1) + j]).
      - The remaining 10 features are used in the entanglement stage: for each qubit j, a controlled rotation gate is
        applied between qubit j and its cyclic neighbor ((j+1) mod n_qubits). The rotation angle is computed as:
            lam * π * ((x[20*(l-1)+10 + (j mod 10)] + x[20*(l-1)+10 + ((j+1) mod 10)])/2),
        where lam is a fixed scaling parameter.
      - CRZ gates are used for odd-numbered layers and CRX gates for even-numbered layers.
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int, lam: float = 1.0) -> None:
        """Initialize the 4-Layer Standard Local Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            lam (float): Fixed scaling parameter for the entanglement gate angles. Default is 1.0.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        self.lam: float = lam
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the feature map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 4 layers of 20 features each.
        """
        expected_length = 4 * 20
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 4 layers
        for l in range(4):
            base = 20 * l
            # Local encoding: apply RY rotations using the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Entanglement stage: apply controlled rotation between neighboring qubits
            for j in range(self.n_qubits):
                index_a = base + 10 + (j % 10)
                index_b = base + 10 + ((j + 1) % 10)
                ent_angle = self.lam * np.pi * ((x[index_a] + x[index_b]) / 2)
                # Use CRZ for odd layers and CRX for even layers
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
