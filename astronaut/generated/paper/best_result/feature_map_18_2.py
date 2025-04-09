import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class AdaptiveSingleAxisWithMidRangeCRotAndISWAPFusionFeatureMap(BaseFeatureMap):
    """
    Adaptive Single-Axis with Mid-Range CRot and ISWAP Fusion Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      
    - Local Encoding:
         The first 10 features are encoded on a 10-qubit register using fixed RY(π * x) rotations,
         ensuring efficient local encoding.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two designated entanglement features are selected from the block starting at index 16*l + 10:
           x_a = x[16*l + 10 + (j mod 6)]
           x_b = x[16*l + 10 + ((j+1) mod 6)]
         The rotation angle is computed as:
           angle = π * (0.5*x_a + 0.5*x_b + 0.1*(x_a - x_b))
         A CRX gate is applied between qubit j and (j+1) mod n_qubits.
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
         For each qubit j, three features are selected at indices (j mod 6), ((j+2) mod 6), and ((j+4) mod 6)
         from the same block. Their average is used to compute the rotation angle (π times the average),
         and a CRY gate is applied between qubit j and (j+2) mod n_qubits.
      
    - Stage 3 (Mid-Range Entanglement):
         For each qubit j, the average of the same two features as in Stage 1 (x_a and x_b) is computed,
         scaled by an adaptive factor λₗ (provided via lambda_factors), yielding an angle:
           angle = π * λₗ * (0.5*x_a + 0.5*x_b)
         A CRot gate (with theta=0.0 and omega=0.0) is applied between qubit j and (j+3) mod n_qubits.
      
    - Stage 4 (ISWAP Fusion):
         To integrate non-nearest neighbor correlations, an ISWAP-like interaction is applied using a parameterized
         IsingXY gate. For j = 0,...,n_qubits/2 - 1 (to avoid duplication), the gate is applied on qubits j and j + n_qubits/2
         with rotation angle π * γ, where γ is a scaling factor.
      
    - Global Entanglement:
         A MultiRZ gate aggregates features from all layers with a rotation angle computed as:
           global_angle = π * Σₗ [ Δₗ * x[16*l+10] ]
         where Δₗ are fixed non-uniform weights.
      
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      lambda_factors (list): A list of 5 adaptive scaling factors for Stage 3. Default is [0.3, 0.3, 0.3, 0.3, 0.3].
      delta_weights (list): A list of 5 weights for global entanglement. Default is [0.15, 0.25, 0.35, 0.15, 0.10].
      gamma (float): Scaling factor for the ISWAP-like fusion interaction. Default is 0.5.
    """
    def __init__(self, n_qubits: int, lambda_factors: list = None, delta_weights: list = None, gamma: float = 0.5) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        # Adaptive scaling factors for mid-range entanglement (Stage 3)
        if lambda_factors is None:
            self.lambda_factors = [0.3, 0.3, 0.3, 0.3, 0.3]
        else:
            self.lambda_factors = lambda_factors
        
        # Global entanglement weights
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
        # Scaling factor for the ISWAP fusion stage (implemented via IsingXY gate)
        self.gamma = gamma
        
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
                angle_immediate = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
                qml.CRX(phi=angle_immediate, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                avg_triple = (x[idx1] + x[idx2] + x[idx3]) / 3.0
                angle_cry = np.pi * avg_triple
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage 3: Mid-Range Entanglement using CRot gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                pair_avg = 0.5 * (x[idx_a] + x[idx_b])
                angle_cret = np.pi * self.lambda_factors[l] * pair_avg
                qml.CRot(phi=angle_cret, theta=0.0, omega=0.0, wires=[j, (j + 3) % self.n_qubits])
            
            # Stage 4: ISWAP Fusion Layer using a parameterized IsingXY gate
            # Apply the gate on pairs to avoid duplication. For 10 qubits, apply on pairs (0,5), (1,6), ..., (4,9).
            for j in range(self.n_qubits // 2):
                qml.IsingXY(phi=np.pi * self.gamma, wires=[j, j + self.n_qubits // 2])
        
        # Global Entanglement: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
