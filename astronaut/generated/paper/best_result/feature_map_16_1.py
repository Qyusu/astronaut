import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DualAxisEncodingWithIssingXXAugmentedEntanglementAndContrastFusionFeatureMap(BaseFeatureMap):
    """
    Dual-Axis Encoding with IssingXX Augmented Entanglement and Contrast Fusion Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      
    - Local Encoding: The first 10 features are encoded onto a 10-qubit register using sequential rotations:
          RZ((π/3) * x[16*(l-1)+j]) followed by RY(π * x[16*(l-1)+j]) for j = 0,...,9.
      
    - Stage 1 (Immediate Neighbor Entanglement with Contrast Fusion):
          For each qubit j, compute from two designated entanglement features (a and b):
             a = x[16*(l-1) + 10 + (j mod 6)]
             b = x[16*(l-1) + 10 + ((j+1) mod 6)]
          Then derive:
             avg  = 0.5 * a + 0.5 * b
             diff = a - b
             angle = π * (avg + α * diff)
          Apply a controlled rotation between qubit j and (j+1) mod n_qubits using:
             - CRZ if the layer is odd
             - CRX if the layer is even
          Additionally, apply an IssingXX gate between the same qubits with rotation angle π * β * diff.
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
          For each qubit j, compute the triple average of features at indices (j mod 6), ((j+2) mod 6), and ((j+4) mod 6)
          from the block starting at 16*(l-1)+10. The rotation angle is π times this average,
          and a CRY gate is applied between qubit j and (j+2) mod n_qubits.
      
    - Stage 3 (Mid-Range Entanglement with Adaptive Calibration):
          For each qubit j, compute the average of two designated entanglement features:
             avg_pair = 0.5*x[16*(l-1)+10+(j mod 6)] + 0.5*x[16*(l-1)+10+((j+1) mod 6)]
          Then, scale this average by an adaptive factor λₗ and apply a CRot gate (with theta=0 and omega=0) between
          qubit j and (j+3) mod n_qubits with rotation angle π * λₗ * avg_pair.
      
    - Global Entanglement:
          A MultiRZ gate is applied across all qubits with rotation angle given by:
             global_angle = π * Σₗ δₗ * x[16*(l-1)+10]
          where δₗ (delta_weights) are non-uniform weights optimized from variance analysis.
    
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      lambda_factors (list): A list of 5 adaptive scaling factors for Stage 3. Default is [0.4, 0.4, 0.4, 0.4, 0.4].
      delta_weights (list): A list of 5 weights for global entanglement. Default is [0.15, 0.25, 0.35, 0.15, 0.10].
      alpha (float): Contrast factor for Stage 1. Default is 0.3.
      beta (float): Scaling factor for the IssingXX gate in Stage 1. Default is 0.4.
    """
    def __init__(self, n_qubits: int, lambda_factors: list = None, delta_weights: list = None, alpha: float = 0.3, beta: float = 0.4) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        # Adaptive scaling factors for Stage 3 (mid-range entanglement)
        if lambda_factors is None:
            self.lambda_factors = [0.4] * 5
        else:
            self.lambda_factors = lambda_factors
        
        # Global entanglement weights
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
        # Contrast and scaling factors for Stage 1
        self.alpha = alpha
        self.beta = beta
        
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Encoding: Apply RZ then RY rotations for qubits 0 through 9
            for j in range(self.n_qubits):
                angle_rz = (np.pi / 3) * x[base + j]
                qml.RZ(phi=angle_rz, wires=j)
                angle_ry = np.pi * x[base + j]
                qml.RY(phi=angle_ry, wires=j)
            
            # Stage 1: Immediate Neighbor Entanglement with Contrast Fusion
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                avg = 0.5 * x[idx_a] + 0.5 * x[idx_b]
                diff = x[idx_a] - x[idx_b]
                angle_ctrl = np.pi * (avg + self.alpha * diff)
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle_ctrl, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle_ctrl, wires=[j, (j + 1) % self.n_qubits])
                
                # IssingXX gate to enhance phase encoding
                angle_isingxx = np.pi * self.beta * diff
                qml.IsingXX(phi=angle_isingxx, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                avg_triple = (x[idx_a] + x[idx_b] + x[idx_c]) / 3.0
                angle_cry = np.pi * avg_triple
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage 3: Mid-Range Entanglement with Adaptive Calibration using CRot gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                avg_pair = 0.5 * x[idx_a] + 0.5 * x[idx_b]
                angle_cret = np.pi * self.lambda_factors[l] * avg_pair
                qml.CRot(phi=angle_cret, theta=0.0, omega=0.0, wires=[j, (j + 3) % self.n_qubits])
        
        # Global Entanglement: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
