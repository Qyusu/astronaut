import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class StreamlinedSingleAxisEncodingWithAdaptiveMidRangeCRotFeatureMap(BaseFeatureMap):
    """
    Streamlined Single-Axis Encoding with Adaptive Mid-Range CRot Feature Map.
    
    This feature map partitions the 80-dimensional input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register using RY(π * x) rotations, which preserves amplitude information.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two designated entanglement features are selected:
           x_a = x[16*(l-1) + 10 + (j mod 6)]
           x_b = x[16*(l-1) + 10 + ((j+1) mod 6)]
         The rotation angle is computed as:
           angle = π * (0.5 * x_a + 0.5 * x_b + 0.15 * (x_a - x_b))
         A controlled rotation is applied between qubit j and (j+1) mod n_qubits using:
           - CRZ if layer (l) is odd (i.e. l+1 odd)
           - CRX if layer is even
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
         For each qubit j, an equal-weight triple average is calculated from the features at indices:
           (j mod 6), ((j+2) mod 6), and ((j+4) mod 6) from the block starting at 16*(l-1)+10.
         The rotation angle is given by π * (average) and a CRY gate is applied between qubit j and (j+2) mod n_qubits.
      
    - Stage 3 (Adaptive Mid-Range Entanglement):
         For each qubit j, the average of the same two features (as in Stage 1) is computed:
           pair_avg = 0.5 * x_a + 0.5 * x_b
         This is scaled by an adaptive factor λₗ (provided via lambda_factors) to yield the rotation angle:
           angle = π * λₗ * (pair_avg)
         A CRot gate (with theta=0.0 and omega=0.0) is applied between qubit j and (j+3) mod n_qubits.
      
    - Global Entanglement:
         A MultiRZ gate is applied over all qubits with rotation angle computed as:
           global_angle = π * Σₗ [ δₗ * x[16*(l-1)+10] ]
         where δₗ (delta_weights) are non-uniform weights.
         
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      lambda_factors (list): A list of 5 adaptive scaling factors for Stage 3. Default is [0.3, 0.3, 0.3, 0.3, 0.3].
      delta_weights (list): A list of 5 weights for global entanglement. Default is [0.15, 0.25, 0.35, 0.15, 0.10].
    """
    def __init__(self, n_qubits: int, lambda_factors: list = None, delta_weights: list = None) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        # Adaptive scaling factors for Stage 3
        if lambda_factors is None:
            self.lambda_factors = [0.3, 0.3, 0.3, 0.3, 0.3]
        else:
            self.lambda_factors = lambda_factors
        
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
            
            # Stage 1: Immediate Neighbor Entanglement
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_immediate = np.pi * (0.5 * x_a + 0.5 * x_b + 0.15 * (x_a - x_b))
                # Use CRZ for odd layers (l+1 odd) and CRX for even layers
                if ((l + 1) % 2) == 1:
                    qml.CRZ(phi=angle_immediate, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle_immediate, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                avg_triple = (x[idx_a] + x[idx_b] + x[idx_c]) / 3.0
                angle_cry = np.pi * avg_triple
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage 3: Adaptive Mid-Range Entanglement using CRot gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                pair_avg = 0.5 * (x[idx_a] + x[idx_b])
                angle_cret = np.pi * self.lambda_factors[l] * pair_avg
                qml.CRot(phi=angle_cret, theta=0.0, omega=0.0, wires=[j, (j + 3) % self.n_qubits])
        
        # Global Entanglement: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
