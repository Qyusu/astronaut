import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class GlobalMultiRZHybridMixedEntanglementFeatureMap(BaseFeatureMap):
    """Global MultiRZ Hybrid Mixed Entanglement Feature Map.
    
    This feature map divides the 80-dimensional input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - The first 10 features are embedded on 10 qubits using RY rotations with angles computed as π * x.
      - The subsequent 6 features are used for intra-layer entanglement:
          For each qubit j:
            * If j is even, the rotation angle is computed as π times the weighted average of two entanglement features
              with weights (0.5, 0.5):
                0.5 * x[16*l + 10 + (j mod 6)] + 0.5 * x[16*l + 10 + ((j+1) mod 6)].
            * If j is odd, the rotation angle is computed as π times the normalized weighted sum of three entanglement features
              with weights (0.3, 0.4, 0.3), divided by 3:
                (0.3 * x[16*l + 10 + (j mod 6)] + 0.4 * x[16*l + 10 + ((j+2) mod 6)] + 0.3 * x[16*l + 10 + ((j+4) mod 6)])/3.
      - The controlled rotation gate used is determined by the layer's parity: CRZ for odd layers and CRX for even layers.
        This gate is applied between qubit j and its cyclic neighbor ((j+1) mod 10).
      - Additionally, each layer applies a fixed CRY gate between qubits 0 and 5 with rotation angle:
            π * ((x[16*l + 11] + x[16*l + 12]) / 2).
    After processing all layers, a final global MultiRZ gate is applied across all 10 qubits.
    Its rotation angle is computed as:
            π * (sum_{l=1}^{5} 0.2 * x[16*(l-1) + 10]),
    capturing inter-layer and long-range correlations.
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Global MultiRZ Hybrid Mixed Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Global MultiRZ Hybrid Mixed Entanglement Feature Map.
        
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
            
            # Intra-layer entanglement: apply controlled rotations with mixed weighted averaging
            for j in range(self.n_qubits):
                if j % 2 == 0:
                    # Even-indexed qubits: use weights (0.5, 0.5)
                    idx1 = base + 10 + (j % 6)
                    idx2 = base + 10 + ((j + 1) % 6)
                    val = 0.5 * x[idx1] + 0.5 * x[idx2]
                else:
                    # Odd-indexed qubits: use weights (0.3, 0.4, 0.3) normalized by 3
                    idx1 = base + 10 + (j % 6)
                    idx2 = base + 10 + ((j + 2) % 6)
                    idx3 = base + 10 + ((j + 4) % 6)
                    val = (0.3 * x[idx1] + 0.4 * x[idx2] + 0.3 * x[idx3]) / 3
                ent_angle = np.pi * val
                # Choose controlled rotation: CRZ for odd layers, CRX for even layers
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
            
            # Apply a fixed CRY gate between qubits 0 and 5 within the layer
            cry_angle = np.pi * ((x[base + 11] + x[base + 12]) / 2)
            qml.CRY(phi=cry_angle, wires=[0, 5])
        
        # Global entanglement: apply a MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += 0.2 * x[base + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
