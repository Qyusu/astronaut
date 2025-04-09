import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DynamicMergedEntanglementWithLinearizedIssingXXAndBFTEnhancementFeatureMap(BaseFeatureMap):
    """
    Dynamic Merged Entanglement with Linearized IssingXX Augmentation and BFT Enhancement Feature Map.
    
    This feature map partitions the 80-dimensional input into 5 layers (each with 16 features).
    For each layer:
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register via RY(π*x) rotations.
      
    - Immediate Neighbor Entanglement:
         For each qubit j, two features are selected from the layer using indices based on (j mod 6) and ((j+1) mod 6).
         A CRX gate is applied between qubit j and (j+1 mod n_qubits) with rotation angle:
             π * (0.5*x_a + 0.5*x_b + 0.1*(x_a - x_b)),
         where x_a and x_b are the selected features (offset by 10).
         Immediately following, an IssingXX augmentation is applied using an IsingXX gate with rotation angle:
             π * gamma * (x_a - x_b),
         implementing a linear correction based on the feature difference.
      
    - Next-Nearest Neighbor Entanglement:
         For each qubit j, three features (indices: (j mod 6), ((j+2) mod 6), ((j+4) mod 6)) are averaged.
         A CRY gate is applied between qubit j and (j+2 mod n_qubits) with rotation angle π times this average.
      
    - Global Fusion:
         Inter-layer features are aggregated by a MultiRZ gate. The rotation angle is computed as
             π * (Σ_l [delta_weights[l] * x_{16*l+10}] + bft),
         where delta_weights are precomputed global weights and bft is an optional Bit Flip Tolerance correction parameter.
      
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits     (int): Number of qubits (ideally 10).
      gamma        (float): Scaling constant for the IssingXX augmentation. Default: 0.1.
      delta_weights (list): Global entanglement weights. Default: [0.15, 0.25, 0.35, 0.15, 0.10].
      bft          (float): Bit Flip Tolerance correction parameter. Default: 0.0.
    """
    def __init__(self, n_qubits: int, 
                 gamma: float = 0.1, 
                 delta_weights: list = None, 
                 bft: float = 0.0) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.gamma = gamma
        self.delta_weights = delta_weights if delta_weights is not None else [0.15, 0.25, 0.35, 0.15, 0.10]
        self.bft = bft
        
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
            
            # Immediate Neighbor Entanglement: CRX followed by linearized IssingXX augmentation
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                # CRX gate
                angle_crx = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
                qml.CRX(phi=angle_crx, wires=[j, (j + 1) % self.n_qubits])
                
                # IssingXX augmentation using IsingXX gate with a linear (non-absolute) difference
                angle_issingxx = np.pi * self.gamma * (x_a - x_b)
                qml.IsingXX(phi=angle_issingxx, wires=[j, (j + 1) % self.n_qubits])
            
            # Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                avg_val = (x[idx1] + x[idx2] + x[idx3]) / 3.0
                angle_cry = np.pi * avg_val
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
        
        # Global Fusion: Aggregate inter-layer features via MultiRZ gate with optional BFT correction
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * x[base + 10]
        # Apply Bit Flip Tolerance correction
        global_angle = np.pi * (global_sum + self.bft)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
