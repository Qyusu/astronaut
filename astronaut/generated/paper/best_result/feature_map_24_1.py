import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class HybridCPSMidRangeWithDynamicAdaptiveGlobalFusionFeatureMap(BaseFeatureMap):
    """
    Hybrid CPS Mid-Range with Dynamic Adaptive Global Fusion Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 0,...,4):
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register via RY(π*x) rotations.
      
    - Stage 1 (Immediate Neighbor Entanglement):
         For each qubit j, two entanglement features are selected from the layer using indices based on (j mod 6) and ((j+1) mod 6).
         A CRX gate is applied between qubit j and (j+1 mod n_qubits) with rotation angle:
             π * (0.5*x_a + 0.5*x_b + 0.1*(x_a - x_b)),
         where x_a and x_b are the selected features from the layer (offset by 10).
      
    - Stage 2 (Next-Nearest Neighbor Entanglement):
         For each qubit j, three features are selected (using indices (j mod 6), ((j+2) mod 6), and ((j+4) mod 6)) and averaged equally.
         A CRY gate is applied between qubit j and (j+2 mod n_qubits) with rotation angle π times the average.
      
    - Stage 3 (Mid-Range Entanglement via CPS):
         For each qubit j, two features (as in Stage 1) are used to compute an angle scaled by an adaptive factor λ_l.
         Here, the adaptive factor is computed via a fixed linear calibration: 
             λ_l = kappa * (F_values[l] - F0),
         where kappa and F0 are constant parameters and F_values is a pre-calibrated vector of noise metrics for each layer.
         A ControlledPhaseShift gate is applied between qubit j and (j+3 mod n_qubits) with rotation angle:
             π * λ_l * (0.5*x_a + 0.5*x_b).
      
    - Global Fusion:
         Inter-layer features are aggregated via a MultiRZ gate with rotation angle computed as
             π * Σ_l [(alpha * S_values[l] + beta) * x_{16*l+10}],
         where S_values is a pre-calibrated vector of separability metrics and alpha, beta are constants.
      
    Note: The input x is expected to have shape (80,).
    
    Parameters:
      n_qubits    (int): Number of qubits (ideally 10).
      kappa       (float): Scaling constant for adaptive mid-range entanglement. Default: 0.3.
      F0          (float): Baseline noise metric constant. Default: 0.5.
      F_values    (list): Pre-calibrated noise metrics for each layer. Default: [1.0, 1.0, 1.0, 1.0, 1.0].
      alpha       (float): Scaling constant for global fusion weights. Default: 0.2.
      beta        (float): Offset constant for global fusion weights. Default: 0.1.
      S_values    (list): Pre-calibrated separability metrics for each layer. Default: [0.0, 0.0, 0.0, 0.0, 0.0].
    """
    def __init__(self, n_qubits: int, 
                 kappa: float = 0.3, 
                 F0: float = 0.5, 
                 F_values: list = None, 
                 alpha: float = 0.2, 
                 beta: float = 0.1, 
                 S_values: list = None) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.kappa = kappa
        self.F0 = F0
        self.F_values = F_values if F_values is not None else [1.0, 1.0, 1.0, 1.0, 1.0]
        self.alpha = alpha
        self.beta = beta
        self.S_values = S_values if S_values is not None else [0.0, 0.0, 0.0, 0.0, 0.0]
        
    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            
            # Local Encoding: Encode first 10 features onto qubits 0 to 9 using RY rotations
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
            
            # Stage 3: Mid-Range Entanglement using ControlledPhaseShift gates
            # Calculate adaptive scaling factor for this layer
            lambda_eff = self.kappa * (self.F_values[l] - self.F0)
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_cps = np.pi * lambda_eff * (0.5 * x_a + 0.5 * x_b)
                qml.ControlledPhaseShift(phi=angle_cps, wires=[j, (j + 3) % self.n_qubits])
        
        # Global Fusion: Aggregate features from all layers via MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            delta = self.alpha * self.S_values[l] + self.beta
            global_sum += delta * x[base + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
