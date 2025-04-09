import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class HybridMixedAveragingEntanglementFeatureMap(BaseFeatureMap):
    """Hybrid Mixed Averaging Entanglement Feature Map.
    
    This feature map divides the 80-dimensional input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - The first 10 features are embedded locally on a 10-qubit register using RY rotations with angles computed as π*x.
      - The remaining 6 features are used in an entanglement stage where the rotation angle is computed differently for even- and odd-indexed qubits:
          • For even-indexed qubits (j even): the angle is π times the average of two entanglement features:
                (x[16*l+10 + (j mod 6)] + x[16*l+10 + ((j+1) mod 6)])/2.
          • For odd-indexed qubits (j odd): the angle is π times the average of three entanglement features:
                (x[16*l+10 + (j mod 6)] + x[16*l+10 + ((j+2) mod 6)] + x[16*l+10 + ((j+4) mod 6)])/3.
      - A controlled rotation is applied between qubit j and its cyclic neighbor (j+1 mod 10).
      - The controlled rotation gate used is CRZ in odd layers and CRX in even layers.
      
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Hybrid Mixed Averaging Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Hybrid Mixed Averaging Entanglement Feature Map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations for the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Entanglement stage: apply controlled rotations with mixed averaging
            for j in range(self.n_qubits):
                if j % 2 == 0:
                    # For even-indexed qubits, average of 2 features
                    idx1 = base + 10 + (j % 6)
                    idx2 = base + 10 + ((j + 1) % 6)
                    theta = (x[idx1] + x[idx2]) / 2
                else:
                    # For odd-indexed qubits, average of 3 features
                    idx1 = base + 10 + (j % 6)
                    idx2 = base + 10 + ((j + 2) % 6)
                    idx3 = base + 10 + ((j + 4) % 6)
                    theta = (x[idx1] + x[idx2] + x[idx3]) / 3
                ent_angle = np.pi * theta
                # Select controlled rotation type based on layer: CRZ for odd layers, CRX for even layers
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])

