import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DynamicDualAxisEncodingWithSimplifiedEntanglementAndAdaptiveGlobalFusionFeatureMap(BaseFeatureMap):
    """
    Dynamic Dual-Axis Encoding with Simplified Entanglement and Adaptive Global Fusion Feature Map.
    
    This feature map partitions the 80-dimensional input into 5 layers (each with 16 features).
    For every layer:
      
    - Local Encoding:
         Each of the first 10 features is embedded onto a 10-qubit register using a dual-axis encoding:
         first an RY(π * x) rotation followed by an RZ((π/3) * x) rotation. This enriches the feature
         representation with minimal overhead.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two entanglement features are selected from the block starting at index 16*l + 10:
           x_a = x[16*l + 10 + (j mod 6)]
           x_b = x[16*l + 10 + ((j+1) mod 6)]
         The rotation angle is computed as:
           angle = π * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
         A CRX gate is applied between qubit j and (j+1) mod n_qubits.
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
         For each qubit j, three features are selected at indices (j mod 6), ((j+2) mod 6), and ((j+4) mod 6)
         from the same block. Their equal-weight average determines the rotation angle (π times the average),
         and a CRY gate is applied between qubit j and (j+2) mod n_qubits.
      
    - Global Entanglement:
         A dynamic MultiRZ gate aggregates inter-layer correlations. Its rotation angle is computed as:
           global_angle = π * Σₗ [ Δₗ * ( x[16*l+10] + η*(x[16*l+10] - x[16*l+11]) ) ]
         where Δₗ (delta_weights) are fixed non-uniform weights and η is a fixed contrast parameter.
      
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      delta_weights (list): A list of 5 weights for global entanglement. Default is [0.15, 0.25, 0.35, 0.15, 0.10].
      eta (float): Contrast parameter for global fusion. Default is 0.1.
    """
    def __init__(self, n_qubits: int, delta_weights: list = None, eta: float = 0.1) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        # Global entanglement weights
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
        # Contrast parameter for adaptive global fusion
        self.eta = eta
        
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Dual-Axis Encoding: Apply RY followed by RZ rotations for qubits 0 through 9
            for j in range(self.n_qubits):
                angle_ry = np.pi * x[base + j]
                qml.RY(phi=angle_ry, wires=j)
                angle_rz = (np.pi / 3) * x[base + j]
                qml.RZ(phi=angle_rz, wires=j)
            
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
        
        # Global Entanglement: Aggregate inter-layer features via a dynamic MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            # Incorporate a contrast term using the first two designated features in the entanglement block
            global_sum += self.delta_weights[l] * (x[base + 10] + self.eta * (x[base + 10] - x[base + 11]))
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
