import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class HybridCPSMidRangeWithDynamicGlobalFusionFeatureMap(BaseFeatureMap):
    """
    Hybrid CPS Mid-Range with Dynamic Global Fusion Feature Map.
    
    This feature map partitions an 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer (l = 0,...,4):
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register using RY(π * x) rotations to preserve local amplitude information.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two auxiliary features are selected from the layer (indices based on (j mod 6) and ((j+1) mod 6), offset by 10).
         A CRX gate is applied between qubit j and (j+1 mod n_qubits) with a rotation angle computed as:
             π * [0.5 * x_a + 0.5 * x_b + ε * (x_a - x_b)]
         where ε (epsilon) is a contrast coefficient that can be adaptively recalibrated.
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
         For each qubit j, three auxiliary features (indices: (j mod 6), ((j+2) mod 6), ((j+4) mod 6), offset by 10) are averaged.
         A CRY gate is applied between qubit j and (j+2 mod n_qubits) with rotation angle π times this average.
      
    - Stage 3 (Mid-Range Entanglement via CPS):
         For each qubit j, using the same pair of auxiliary features as Stage 1, a mid-range entanglement is applied using a ControlledPhaseShift (CPS) gate.
         The rotation angle is computed as:
             π * λₗ * (0.5 * x_a + 0.5 * x_b),
         where the adaptive scaling factor λₗ is determined for each layer via a fixed linear calibration:
             λₗ = κ * (Fₗ - F₀),
         with κ (kappa), F₀, and Fₗ (elements of F_values) provided as parameters.
      
    - Global Fusion:
         After processing all layers, a MultiRZ gate aggregates inter-layer features.
         The rotation angle is set to:
             π * Σₗ [δₗ * x_{16*l+10}],
         where δₗ (delta_weights) are dynamically adjusted global weights.
      
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits     (int): Number of qubits (ideally 10).
      kappa        (float): Scaling constant for mid-range entanglement. Default: 0.3.
      F0           (float): Baseline noise metric constant. Default: 0.5.
      F_values     (list): Pre-calibrated noise metrics for each layer. Default: [1.0, 1.0, 1.0, 1.0, 1.0].
      epsilon      (float): Contrast coefficient for CRX entanglement. Default: 0.1.
      delta_weights(list): Global fusion weights. Default: [0.15, 0.25, 0.35, 0.15, 0.10].
    """
    def __init__(self, n_qubits: int,
                 kappa: float = 0.3,
                 F0: float = 0.5,
                 F_values: list = None,
                 epsilon: float = 0.1,
                 delta_weights: list = None) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.kappa = kappa
        self.F0 = F0
        self.F_values = F_values if F_values is not None else [1.0, 1.0, 1.0, 1.0, 1.0]
        self.epsilon = epsilon
        self.delta_weights = delta_weights if delta_weights is not None else [0.15, 0.25, 0.35, 0.15, 0.10]
        
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Encoding: Apply RY rotations for qubits 0 to 9 (first 10 features)
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Stage 1: Immediate Neighbor Entanglement using CRX gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_crx = np.pi * (0.5 * x_a + 0.5 * x_b + self.epsilon * (x_a - x_b))
                qml.CRX(phi=angle_crx, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                avg_val = (x[idx1] + x[idx2] + x[idx3]) / 3.0
                angle_cry = np.pi * avg_val
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage 3: Mid-Range Entanglement using CPS (ControlledPhaseShift) gates
            # Compute the adaptive scaling factor for this layer
            lambda_eff = self.kappa * (self.F_values[l] - self.F0)
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_cps = np.pi * lambda_eff * (0.5 * x_a + 0.5 * x_b)
                qml.ControlledPhaseShift(phi=angle_cps, wires=[j, (j + 3) % self.n_qubits])
        
        # Global Fusion: Aggregate inter-layer features via MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * x[base + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
