import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DualAxisLocalEncodingWithHybridCPSEntanglementAndDynamicGlobalFusionFeatureMap(BaseFeatureMap):
    """
    Dual-Axis Local Encoding with Hybrid ControlledPhaseShift Entanglement and Dynamic Global Fusion Feature Map.
    
    This feature map partitions an 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer (l = 0,...,4):
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register using fixed RY(π*x) rotations to capture amplitude information,
         followed by RZ((π/3)*x) rotations to introduce an orthogonal phase encoding, yielding a balanced dual-axis representation.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two auxiliary entanglement features are selected (indices based on (j mod 6) and ((j+1) mod 6), offset by 10).
         A CRX gate is applied between qubit j and its immediate neighbor with a rotation angle computed as:
             π * (0.5*x_a + 0.5*x_b + ε*(x_a - x_b)),
         where ε is a modest contrast coefficient (default 0.05) that can be adaptively recalibrated via linear noise metrics.
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
         For each qubit j, three auxiliary features (indices: (j mod 6), ((j+2) mod 6), ((j+4) mod 6), offset by 10) are averaged.
         A CRY gate is applied between qubit j and the qubit two positions ahead with rotation angle π times the average value.
      
    - Stage 3 (Mid-Range Entanglement via CPS):
         For each qubit j, using the same pair of auxiliary features as in Stage 1, a ControlledPhaseShift (CPS) gate is applied between qubit j and
         the qubit three positions ahead. The rotation angle is given by:
             π * λₗ * (0.5*x_a + 0.5*x_b),
         where λₗ is an adaptive scaling factor for the layer provided via lambda_values.
      
    - Global Fusion:
         After processing all layers, a MultiRZ gate aggregates inter-layer features with a rotation angle:
             π * Σₗ (δₗ * x_{16*l+10}),
         where δₗ (delta_weights) are dynamically adapted global fusion weights.
      
    Note: The input x is expected to have shape (80,).
      
    Parameters:
      n_qubits     (int): Number of qubits (ideally 10).
      epsilon      (float): Contrast coefficient for CRX entanglement. Default: 0.05.
      lambda_values (list): Adaptive scaling factors for CPS entanglement per layer. Default: [1.0, 1.0, 1.0, 1.0, 1.0].
      delta_weights (list): Global fusion weights. Default: [0.15, 0.25, 0.35, 0.15, 0.10].
    """
    def __init__(self, n_qubits: int, epsilon: float = 0.05, lambda_values: list = None, delta_weights: list = None) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.epsilon = epsilon
        self.lambda_values = lambda_values if lambda_values is not None else [1.0, 1.0, 1.0, 1.0, 1.0]
        self.delta_weights = delta_weights if delta_weights is not None else [0.15, 0.25, 0.35, 0.15, 0.10]
        
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Encoding: Apply RY and RZ rotations for qubits 0 to 9
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
            
            # Stage 3: Mid-Range Entanglement using ControlledPhaseShift gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_cps = np.pi * self.lambda_values[l] * (0.5 * x_a + 0.5 * x_b)
                qml.ControlledPhaseShift(phi=angle_cps, wires=[j, (j + 3) % self.n_qubits])
        
        # Global Fusion: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * x[base + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
