import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DualStageIntraLayerEntanglementFeatureMap(BaseFeatureMap):
    """Dual-Stage Intra-Layer Entanglement Feature Map.
    
    This feature map divides the 80-dimensional input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - The first 10 features are encoded locally on 10 qubits via RY rotations with angles computed as π*x.
      - The remaining 6 features are used in a two-stage entanglement process within each layer:
          Stage 1 (nearest-neighbor entanglement):
            * For each qubit j, compute the rotation angle as π times the average of two features:
                (x[16*l+10 + (j mod 6)] + x[16*l+10 + ((j+1) mod 6)])/2,
              and apply a controlled rotation between qubit j and (j+1 mod 10).
          Stage 2 (next-nearest neighbor entanglement):
            * For each qubit j, compute the rotation angle as π times the average of three features:
                (x[16*l+10 + (j mod 6)] + x[16*l+10 + ((j+2) mod 6)] + x[16*l+10 + ((j+4) mod 6)])/3,
              and apply a controlled rotation between qubit j and (j+2 mod 10).
      - Both stages use the same type of controlled rotation gate per layer: CRZ if the layer is odd, CRX if even.
      
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Dual-Stage Intra-Layer Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Dual-Stage Intra-Layer Entanglement Feature Map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations for the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Stage 1: Nearest-neighbor entanglement
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 1) % 6)
                theta1 = (x[idx1] + x[idx2]) / 2
                angle1 = np.pi * theta1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                theta2 = (x[idx1] + x[idx2] + x[idx3]) / 3
                angle2 = np.pi * theta2
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
                else:
                    qml.CRX(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
