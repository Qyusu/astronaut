import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class SimplifiedAlternatingEntanglementFeatureMap(BaseFeatureMap):
    """Simplified Alternating Entanglement Feature Map.
    
    This feature map partitions the 80-dimensional input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - The first 10 features are embedded into the quantum state via single-qubit rotations,
        applying RY(gamma * x) on each of the 10 qubits, where gamma is a fixed scaling factor.
      - The remaining 6 features are used in the entanglement stage: for each qubit j, a controlled rotation
        is applied between qubit j and its cyclic neighbor ((j+1) mod n_qubits), with the rotation angle computed
        as gamma multiplied by the average of two features
            (x[16*(l-1)+10 + (j mod 6)] + x[16*(l-1)+10 + ((j+1) mod 6)])/2.
      - Additionally, odd-numbered layers use CRZ gates for entanglement, while even-numbered layers use CRX gates.
    
    Recommended fixed scaling factor: gamma = π/4.
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int, gamma: float = np.pi/4) -> None:
        """Initialize the Simplified Alternating Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            gamma (float): Fixed scaling factor for rotation angles. Default is π/4.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        self.gamma: float = gamma
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the feature map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations using the first 10 features
            for j in range(self.n_qubits):
                angle = self.gamma * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Entanglement stage: apply controlled rotation between neighboring qubits
            for j in range(self.n_qubits):
                index_a = base + 10 + (j % 6)
                index_b = base + 10 + ((j + 1) % 6)
                ent_angle = self.gamma * ((x[index_a] + x[index_b]) / 2)
                # Use CRZ for odd layers and CRX for even layers
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
