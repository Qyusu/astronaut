import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class MergedImmediateAndNextNearestEntanglementWithOptimizedGlobalAggregationFeatureMap(BaseFeatureMap):
    """
    Merged Immediate and Next-Nearest Entanglement with Optimized Global Aggregation Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 0,...,4):
      - Local Encoding: The first 10 features are embedded onto a 10-qubit register via RY rotations:
            RY(π * x[16*l + j]) for j = 0,...,9.
      - Stage 1 (Immediate Neighbor Entanglement):
            For each qubit j, compute the rotation angle as
              θ₁ = w_stage1[l][j] * x[16*l + 10 + (j mod 6)] + (1 - w_stage1[l][j]) * x[16*l + 10 + ((j+1) mod 6)],
            and apply a controlled rotation between qubit j and (j+1) mod n_qubits using:
              - CRZ if (l+1) is odd
              - CRX if (l+1) is even
      - Stage 2 (Next-Nearest Neighbor Entanglement):
            For each qubit j, compute the rotation angle as
              θ₂ = v_stage2[l][j][0] * x[16*l + 10 + (j mod 6)] +
                    v_stage2[l][j][1] * x[16*l + 10 + ((j+2) mod 6)] +
                    v_stage2[l][j][2] * x[16*l + 10 + ((j+4) mod 6)],
            and apply a CRY gate between qubit j and (j+2) mod n_qubits with rotation angle π·θ₂.
      - Global Entanglement: A MultiRZ gate is applied across all qubits with rotation angle
            π * (Σₗ δₗ * x[16*l + 10]),
            where δₗ are non-uniform weights derived offline.
    
    Note: The input x is expected to have shape (80,).
    """
    def __init__(self, n_qubits: int, w_stage1: list = None, v_stage2: list = None, delta_weights: list = None) -> None:
        """
        Initialize the Merged Immediate and Next-Nearest Entanglement with Optimized Global Aggregation Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            w_stage1 (list): 5x10 weight matrix for Stage 1 (default: all 0.5).
            v_stage2 (list): 5x10x3 weight tensor for Stage 2 (default: all 1/3).
            delta_weights (list): List of 5 weights for global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        if w_stage1 is None:
            self.w_stage1 = [[0.5 for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.w_stage1 = w_stage1
        
        if v_stage2 is None:
            self.v_stage2 = [[[1/3 for _ in range(3)] for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.v_stage2 = v_stage2
        
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
    def feature_map(self, x: np.ndarray) -> None:
        """
        Create the quantum circuit for the Merged Immediate and Next-Nearest Entanglement with Optimized Global Aggregation Feature Map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations for the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Stage 1: Immediate neighbor entanglement
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                theta1 = self.w_stage1[l][j] * x[idx_a] + (1 - self.w_stage1[l][j]) * x[idx_b]
                angle1 = np.pi * theta1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                theta2 = (self.v_stage2[l][j][0] * x[idx_a] +
                          self.v_stage2[l][j][1] * x[idx_b] +
                          self.v_stage2[l][j][2] * x[idx_c])
                angle2 = np.pi * theta2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
        
        # Global entanglement: apply MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
