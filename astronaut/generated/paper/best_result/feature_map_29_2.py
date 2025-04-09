import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DynamicMergedEntanglementWithRealTimeCalibratedIssingXXAugmentationFeatureMap(BaseFeatureMap):
    """
    Dynamic Merged Entanglement with Real-Time Calibrated IssingXX Augmentation Feature Map.
    
    This feature map partitions an 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer (l = 0,...,4):
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register using fixed RY(π*x) rotations to ensure robust local amplitude preservation.
      
    - Immediate Neighbor Entanglement:
         For each qubit j, two entanglement features are selected based on (j mod 6) and ((j+1) mod 6), offset by 10.
         A CRX gate is applied between qubit j and its immediate neighbor with rotation angle:
             π * (0.5*x_a + 0.5*x_b + 0.1*(x_a - x_b)),
         where the contrast term 0.1 is fixed.
      
    - Next-Nearest Neighbor Entanglement:
         For each qubit j, three auxiliary features (indices: (j mod 6), ((j+2) mod 6), ((j+4) mod 6), offset by 10) are averaged.
         A CRY gate is applied between qubit j and the qubit two positions ahead with rotation angle π times the average value.
      
    - IssingXX Augmentation:
         On the same qubit pairs as used in the immediate neighbor entanglement, an IssingXX gate is applied.
         The rotation angle is computed as:
             π * γ * (x_a - x_b),
         where γ is an adaptive scaling factor (default 0.1) calibrated in real time.
      
    - Global Fusion:
         After processing all layers, a MultiRZ gate aggregates inter-layer features with a rotation angle:
             π * Σₗ (Δₗ * x_{16*l+10}),
         where Δₗ (delta_weights) are adaptive global fusion weights.
      
    Note: The input x is expected to have shape (80,).
      
    Parameters:
      n_qubits  (int): Number of qubits (ideally 10).
      gamma     (float): Real-time calibrated scaling constant for the IssingXX augmentation. Default: 0.1.
      delta_weights (list): Global fusion weights. Default: [0.15, 0.25, 0.35, 0.15, 0.10].
    """
    def __init__(self, n_qubits: int, gamma: float = 0.1, delta_weights: list = None) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.delta_weights = delta_weights if delta_weights is not None else [0.15, 0.25, 0.35, 0.15, 0.10]
        
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Encoding: Apply RY rotations for qubits 0 to 9
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Immediate Neighbor Entanglement: CRX gate
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_crx = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
                qml.CRX(phi=angle_crx, wires=[j, (j + 1) % self.n_qubits])
            
            # Next-Nearest Neighbor Entanglement: CRY gate
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                avg_val = (x[idx1] + x[idx2] + x[idx3]) / 3.0
                angle_cry = np.pi * avg_val
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
            
            # IssingXX Augmentation on immediate neighbor pairs
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_issingxx = np.pi * self.gamma * (x_a - x_b)
                qml.IsingXX(phi=angle_issingxx, wires=[j, (j + 1) % self.n_qubits])
        
        # Global Fusion: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * x[base + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
