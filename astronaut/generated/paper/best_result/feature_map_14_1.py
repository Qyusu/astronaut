import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class AdaptiveMultiStageEntanglementFeatureMapWithDataDrivenMidRangeCalibrationFeatureMap(BaseFeatureMap):
    """
    Adaptive Multi-Stage Entanglement Feature Map with Data-Driven Mid-Range Calibration.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - Local Encoding: The first 10 features are embedded onto a 10-qubit register via RY rotations:
            RY(π * x[16*(l-1) + j]) for j = 0,...,9.
      - Stage 1 (Immediate Neighbor Entanglement):
            For each qubit j, the rotation angle is computed as a weighted pairwise average:
              θ₁ = w_stage1[l][j] * x[16*(l-1) + 10 + (j mod 6)] + (1 - w_stage1[l][j]) * x[16*(l-1) + 10 + ((j+1) mod 6)],
            and applied using a controlled rotation with:
              - CRZ if the layer (l) is odd
              - CRX if the layer (l) is even
      - Stage 2 (Next-Nearest Neighbor Entanglement):
            For each qubit j, the rotation angle is computed using an equal-weight triple average:
              θ₂ = (1/3) * (x[16*(l-1) + 10 + (j mod 6)] + x[16*(l-1) + 10 + ((j+2) mod 6)] + x[16*(l-1) + 10 + ((j+4) mod 6)]),
            and applied via a CRY gate between qubit j and (j+2) mod n_qubits with rotation angle π·θ₂.
      - Stage 3 (Mid-Range Entanglement with Adaptive Calibration):
            For each qubit j, the rotation angle is computed as:
              θ₃ = 0.5 * x[16*(l-1) + 10 + (j mod 6)] + 0.5 * x[16*(l-1) + 10 + ((j+1) mod 6)],
            then scaled by an adaptive factor λₗ (lambda_factors[l]) derived from offline calibration,
            and applied using a CRot gate between qubit j and (j+3) mod n_qubits with rotation angle π·λₗ·θ₃.
      - Global Entanglement: A MultiRZ gate is applied across all qubits with rotation angle
            π * (Σₗ δₗ * x[16*(l-1) + 10]),
            where δₗ (delta_weights) are non-uniform weights optimized from variance analysis.
    
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      w_stage1 (list): A 5x10 weight matrix for Stage 1. Default is a 5x10 matrix with all values 0.5.
      lambda_factors (list): A list of 5 adaptive scaling factors for Stage 3. Default is [0.4, 0.4, 0.4, 0.4, 0.4].
      delta_weights (list): A list of 5 weights for global entanglement. Default is [0.15, 0.25, 0.35, 0.15, 0.10].
    """
    def __init__(self, n_qubits: int, w_stage1: list = None, lambda_factors: list = None, delta_weights: list = None) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        # Initialize weight matrix for Stage 1 (immediate neighbor entanglement)
        if w_stage1 is None:
            # Create a 5x(n_qubits) matrix with default value 0.5
            self.w_stage1 = [[0.5 for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.w_stage1 = w_stage1
        
        # Adaptive scaling factors for Stage 3 (mid-range entanglement)
        if lambda_factors is None:
            self.lambda_factors = [0.4 for _ in range(5)]
        else:
            self.lambda_factors = lambda_factors
        
        # Non-uniform global weights for the MultiRZ gate
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
            # Local Encoding: Apply RY rotations for the first 10 features of the layer
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Stage 1: Immediate Neighbor Entanglement with adaptive weights
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                theta1 = self.w_stage1[l][j] * x[idx_a] + (1 - self.w_stage1[l][j]) * x[idx_b]
                angle1 = np.pi * theta1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                theta2 = (1/3) * (x[idx_a] + x[idx_b] + x[idx_c])
                angle2 = np.pi * theta2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage 3: Mid-Range Entanglement with Adaptive Calibration using CRot gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                theta3 = 0.5 * x[idx_a] + 0.5 * x[idx_b]
                angle3 = np.pi * self.lambda_factors[l] * theta3
                qml.CRot(phi=angle3, theta=0.0, omega=0.0, wires=[j, (j + 3) % self.n_qubits])
        
        # Global Entanglement: Apply MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
