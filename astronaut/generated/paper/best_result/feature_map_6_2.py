import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DualStageIntraLayerWithGlobalMultiRZEntanglementFeatureMap(BaseFeatureMap):
    """Dual-Stage Intra-Layer with Global MultiRZ Entanglement Feature Map.
    
    This feature map divides the 80-dimensional input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - The first 10 features are encoded locally on 10 qubits via RY rotations with angles computed as π * x.
      - Stage 1 (immediate neighbor entanglement):
          For each qubit j, a controlled rotation is applied between qubit j and its cyclic neighbor ((j+1) mod 10).
          The rotation angle is computed as π times the weighted sum of two entanglement features with weights (0.6, 0.4):
              0.6 * x[16*l + 10 + (j mod 6)] + 0.4 * x[16*l + 10 + ((j+1) mod 6)].
          The controlled rotation gate used is CRZ for odd layers and CRX for even layers.
      - Stage 2 (next-nearest neighbor entanglement):
          For each qubit j, a CRY gate is applied between qubit j and qubit ((j+2) mod 10).
          The rotation angle is computed as π times the weighted sum of three entanglement features with weights (0.3, 0.4, 0.3):
              0.3 * x[16*l + 10 + (j mod 6)] + 0.4 * x[16*l + 10 + ((j+2) mod 6)] + 0.3 * x[16*l + 10 + ((j+4) mod 6)].
    After processing all layers, a final global MultiRZ gate is applied across all 10 qubits with rotation angle
          π * (sum_{l=1}^{5} 0.2 * x[16*(l-1) + 10]),
    capturing inter-layer correlations.
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Dual-Stage Intra-Layer with Global MultiRZ Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Dual-Stage Intra-Layer with Global MultiRZ Entanglement Feature Map.
        
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
            
            # Stage 1: Immediate neighbor entanglement
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 1) % 6)
                weighted_val = 0.6 * x[idx1] + 0.4 * x[idx2]
                angle1 = np.pi * weighted_val
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                weighted_val2 = 0.3 * x[idx1] + 0.4 * x[idx2] + 0.3 * x[idx3]
                angle2 = np.pi * weighted_val2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
        
        # Global entanglement: apply a MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += 0.2 * x[base + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
