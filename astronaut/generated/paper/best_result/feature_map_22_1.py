import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class HybridNestedCRotAndIssingXXEntanglementFeatureMap(BaseFeatureMap):
    """
    Hybrid Nested CRot and IssingXX Entanglement Feature Map.

    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer:
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register via RY(π * x) rotations, preserving amplitude information.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two features (x_a and x_b) are selected from positions based on (j mod 6) and ((j+1) mod 6) within the layer.
         The rotation angle for a CRX gate is computed as:
             angle = π * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
         and applied between qubit j and its immediate neighbor ((j+1) mod n_qubits).
      
    - Stage 2 (Nested CRot Entanglement):
         A nested CRot gate is applied between qubit j and the qubit at (j+3) mod n_qubits.
         Its rotation angle is given by:
             angle = π * λₗ * (0.5 * x_a + 0.5 * x_b)
         where λₗ is a layer-specific adaptive scaling factor provided via lambda_factors.
      
    - Stage 3 (IssingXX Entanglement):
         An IsingXX gate (serving as the IssingXX gate) is applied between qubit j and its neighbor ((j+1) mod n_qubits) with a rotation angle:
             angle = π * β * (x_a - x_b)
         where β is a fixed linear coefficient.
      
    - Global Entanglement:
         Inter-layer correlations are aggregated via a MultiRZ gate with rotation angle computed as:
             global_angle = π * Σₗ [ δₗ * x_{16*l+10} ]
         where δₗ are externally calibrated non-uniform weights provided via delta_weights.
      
    Note: The input x is expected to have shape (80,).
      
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      lambda_factors (list): Adaptive scaling factors for the nested CRot entanglement. Default: [0.3, 0.3, 0.3, 0.3, 0.3].
      delta_weights (list): Global entanglement weights. Default: [0.15, 0.25, 0.35, 0.15, 0.10].
      beta (float): Scaling parameter for the IssingXX entanglement stage. Default: 0.1.
    """
    def __init__(self, n_qubits: int, lambda_factors: list = None, delta_weights: list = None, beta: float = 0.1) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.lambda_factors = lambda_factors if lambda_factors is not None else [0.3, 0.3, 0.3, 0.3, 0.3]
        self.delta_weights = delta_weights if delta_weights is not None else [0.15, 0.25, 0.35, 0.15, 0.10]
        self.beta = beta
    
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Encoding: Apply RY rotations for qubits 0 through 9
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Stage 1: Immediate Neighbor Entanglement using CRX gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_crx = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
                qml.CRX(phi=angle_crx, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Nested CRot Entanglement
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                # Compute the rotation angle using the adaptive scaling factor for this layer
                angle_nested = np.pi * self.lambda_factors[l] * (0.5 * x[idx_a] + 0.5 * x[idx_b])
                qml.CRot(phi=angle_nested, theta=0.0, omega=0.0, wires=[j, (j + 3) % self.n_qubits])
            
            # Stage 3: IssingXX Entanglement using IsingXX gate
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                angle_issingxx = np.pi * self.beta * (x[idx_a] - x[idx_b])
                qml.IsingXX(phi=angle_issingxx, wires=[j, (j + 1) % self.n_qubits])
        
        # Global Entanglement: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * x[base + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
