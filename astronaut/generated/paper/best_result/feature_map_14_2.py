import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class StreamlinedDualStageWithUnifiedAdaptiveGlobalFusionFeatureMap(BaseFeatureMap):
    """
    Streamlined Dual-Stage with Unified Adaptive Global Fusion Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - Local Encoding: The first 10 features are embedded onto a 10-qubit register via RY rotations:
            RY(π * x[16*(l-1) + j]) for j = 0,...,9.
      - Stage A (Immediate Neighbor Entanglement):
            For each qubit j, compute the rotation angle as
              φ₁ = 0.5 * x[16*(l-1) + 10 + (j mod 6)] + 0.5 * x[16*(l-1) + 10 + ((j+1) mod 6)],
            and apply a controlled rotation between qubit j and (j+1) mod n_qubits using:
              - CRZ if the layer is odd
              - CRX if the layer is even
      - Stage B (Next-Nearest Neighbor Entanglement):
            For each qubit j, compute the rotation angle as
              φ₂ = (1/3) * (x[16*(l-1) + 10 + (j mod 6)] + x[16*(l-1) + 10 + ((j+2) mod 6)] + x[16*(l-1) + 10 + ((j+4) mod 6)]),
            and apply a CRY gate between qubit j and (j+2) mod n_qubits with rotation angle π·φ₂.
      - Global Entanglement (Unified Adaptive Global Fusion):
            A MultiRZ gate is applied across all qubits with rotation angle
              π * (Σₗ Δₗ * x[16*(l-1) + 10]),
            where Δₗ are adaptive non-uniform weights optimized from separability and quantum Fisher information metrics.
    
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      delta_weights (list): A list of 5 adaptive global fusion weights. Default is [0.15, 0.25, 0.35, 0.15, 0.10].
    """
    def __init__(self, n_qubits: int, delta_weights: list = None) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        # Global fusion weights for the MultiRZ gate
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
            # Local Encoding: Apply RY rotations for the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Stage A: Immediate Neighbor Entanglement
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                phi1 = 0.5 * x[idx_a] + 0.5 * x[idx_b]
                angle1 = np.pi * phi1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage B: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                phi2 = (1/3) * (x[idx_a] + x[idx_b] + x[idx_c])
                angle2 = np.pi * phi2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
        
        # Global Entanglement: Apply MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
