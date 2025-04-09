import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class StreamlinedSingleAxisWithSimplifiedEntanglementAndAdaptiveGlobalFusionFeatureMap(BaseFeatureMap):
    """
    Streamlined Single-Axis with Simplified Entanglement and Adaptive Global Fusion Feature Map.
    
    This feature map partitions the 80-dimensional input into 5 layers (each with 16 features).
    For each layer:
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register using RY(π * x) rotations,
         preserving essential amplitude information.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two entanglement features are selected from the block starting at index 16*l + 10:
           x_a = x[16*l + 10 + (j mod 6)]
           x_b = x[16*l + 10 + ((j+1) mod 6)]
         The rotation angle is computed as:
           angle = π * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
         A CRX gate is applied between qubit j and (j+1) mod n_qubits.
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
         For each qubit j, three features are selected from indices (j mod 6), ((j+2) mod 6), and ((j+4) mod 6)
         within the same layer. Their average determines the rotation angle (π times the average),
         and a CRY gate is applied between qubit j and (j+2) mod n_qubits.
      
    - Global Entanglement:
         Inter-layer correlations are aggregated via a MultiRZ gate whose rotation angle is given by:
           global_angle = π * Σₗ [ δₗ * x[16*l+10] ]
         where δₗ are fixed non-uniform weights obtained from offline calibration.
      
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      delta_weights (list): A list of 5 weights for global entanglement. Default is [0.15, 0.25, 0.35, 0.15, 0.10].
    """
    def __init__(self, n_qubits: int, delta_weights: list = None) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        # Global entanglement weights
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Encoding: Apply RY rotations for qubits 0 through 9
            for j in range(self.n_qubits):
                angle_ry = np.pi * x[base + j]
                qml.RY(phi=angle_ry, wires=j)
            
            # Stage 1: Immediate Neighbor Entanglement using CRX gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_crx = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
                qml.CRX(phi=angle_crx, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                avg_val = (x[idx1] + x[idx2] + x[idx3]) / 3.0
                angle_cry = np.pi * avg_val
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
        
        # Global Entanglement: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * x[base + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
