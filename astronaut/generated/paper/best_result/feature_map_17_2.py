import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class EnhancedDualAxisEncodingWithRecalibratedContrastAndSimplifiedEntanglementFeatureMap(BaseFeatureMap):
    """
    Enhanced Dual-Axis Encoding with Recalibrated Contrast and Simplified Entanglement Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register using a dual-axis process:
         an initial RY(π * x) rotation followed by an RZ((π/3) * x) rotation for each qubit (j = 0,...,9).
      
    - Stage A (Immediate Neighbor Entanglement):
         For each qubit j, two designated entanglement features are selected:
           x_a = x[16*(l-1) + 10 + (j mod 6)]
           x_b = x[16*(l-1) + 10 + ((j+1) mod 6)]
         The rotation angle is computed as:
           angle = π * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
         A controlled rotation is applied between qubit j and (j+1) mod n_qubits using:
           - CRZ if layer (l) is odd
           - CRX if layer is even
      
    - Stage B (Next-Nearest Neighbor Entanglement):
         For each qubit j, an equal-weight triple average is computed from the features at indices
         (j mod 6), ((j+2) mod 6), and ((j+4) mod 6) from the block starting at 16*(l-1)+10.
         The rotation angle is given by π times this average, and a CRY gate is applied between qubit j
         and (j+2) mod n_qubits.
      
    - Global Entanglement:
         A MultiRZ gate aggregates features across layers.
         Here a fixed uniform weight of 1/5 is used for each layer, leading to a rotation angle:
           global_angle = π * Σₗ [(1/5) * x[16*(l-1)+10]]
    
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
    """
    def __init__(self, n_qubits: int) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Encoding: Apply RY followed by RZ rotations for qubits 0 through 9
            for j in range(self.n_qubits):
                angle_ry = np.pi * x[base + j]
                qml.RY(phi=angle_ry, wires=j)
                angle_rz = (np.pi / 3) * x[base + j]
                qml.RZ(phi=angle_rz, wires=j)
            
            # Stage A: Immediate Neighbor Entanglement
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_immediate = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
                if ((l + 1) % 2) == 1:
                    qml.CRZ(phi=angle_immediate, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle_immediate, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage B: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                avg_triple = (x[idx_a] + x[idx_b] + x[idx_c]) / 3.0
                angle_cry = np.pi * avg_triple
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
        
        # Global Entanglement: Aggregate features via a MultiRZ gate using uniform weights
        global_sum = 0.0
        for l in range(5):
            global_sum += (1/5) * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
