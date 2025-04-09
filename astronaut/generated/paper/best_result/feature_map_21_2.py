import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DynamicSingleAxisWithFullThreeStageEntanglementAndContrastEnhancedGlobalFusionFeatureMap(BaseFeatureMap):
    """
    Dynamic Single-Axis with Full Three-Stage Entanglement and Contrast-Enhanced Global Fusion Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer:
      
    - Local Encoding:
         The first 10 features are mapped onto a 10-qubit register via RY(π * x) rotations, ensuring a stable, linear encoding.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two features from the block starting at index 16*l + 10 are used:
             x_a = x[16*l + 10 + (j mod 6)]
             x_b = x[16*l + 10 + ((j+1) mod 6)]
         The rotation angle is computed as:
             angle = π * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
         A CRX gate is applied between qubit j and (j+1) mod n_qubits.
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
         For each qubit j, three features (from indices (j mod 6), ((j+2) mod 6), and ((j+4) mod 6)) are averaged,
         and the resulting value multiplied by π determines the rotation angle for a CRY gate applied between qubit j and (j+2) mod n_qubits.
      
    - Stage 3 (Mid-Range Entanglement):
         For each qubit j, a CRot gate reintroduces mid-range correlations by acting on two features from the immediate
         entanglement block. The angle is given by π times an adaptive scaling factor λₗ (from lambda_factors)
         multiplied by the half-sum of the two selected features, coupling qubit j and (j+3) mod n_qubits.
      
    - Global Fusion:
         Outputs from all layers are aggregated via a MultiRZ gate with rotation angle computed as:
             global_angle = π * Σₗ [ Δₗ * ( x[16*l+10] + η * (x[16*l+10] - x[16*l+11]) ) ]
         where Δₗ are the global entanglement weights (delta_weights) and η is a fixed contrast parameter.
      
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      lambda_factors (list): Adaptive scaling factors for mid-range entanglement. Default: [0.3, 0.3, 0.3, 0.3, 0.3].
      delta_weights (list): Global entanglement weights. Default: [0.15, 0.25, 0.35, 0.15, 0.10].
      eta (float): Contrast parameter for global fusion. Default: 0.1.
    """
    def __init__(self, n_qubits: int, lambda_factors: list = None, delta_weights: list = None, eta: float = 0.1) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.lambda_factors = lambda_factors if lambda_factors is not None else [0.3, 0.3, 0.3, 0.3, 0.3]
        self.delta_weights = delta_weights if delta_weights is not None else [0.15, 0.25, 0.35, 0.15, 0.10]
        self.eta = eta
    
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
            
            # Stage 2: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                avg_val = (x[idx1] + x[idx2] + x[idx3]) / 3.0
                angle_cry = np.pi * avg_val
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage 3: Mid-Range Entanglement using CRot gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                angle_crot = np.pi * self.lambda_factors[l] * (0.5 * x[idx_a] + 0.5 * x[idx_b])
                qml.CRot(phi=angle_crot, theta=0.0, omega=0.0, wires=[j, (j + 3) % self.n_qubits])
        
        # Global Fusion: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * (x[base + 10] + self.eta * (x[base + 10] - x[base + 11]))
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
