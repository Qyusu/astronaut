import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DynamicMergedEntanglementWithControlledPhaseShiftFusionFeatureMap(BaseFeatureMap):
    """
    Dynamic Merged Entanglement with ControlledPhaseShift Fusion Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer:
      
    - Local Encoding:
         The first 10 features are encoded onto a 10-qubit register via RY(π * x) rotations, ensuring a consistent linear mapping.
      
    - Merged Entanglement Stage:
         Immediate neighbor and mid-range couplings are merged using ControlledPhaseShift gates.
         For each qubit j, two features are selected from the layer (using indices based on (j mod 6) and ((j+1) mod 6)).
         The rotation angle for the CPS gate is computed as:
             angle = π * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
         and applied between qubit j and ((j+1) mod n_qubits).
      
    - Extended Entanglement:
         Next-nearest neighbor entanglement is achieved via CRY gates. For each qubit j, three features are averaged:
             avg = (x[idx1] + x[idx2] + x[idx3]) / 3
         and the rotation angle is set as π * avg, applied between qubit j and ((j+2) mod n_qubits).
      
    - Global Fusion:
         Inter-layer features are aggregated by a MultiRZ gate whose rotation angle is computed as:
             global_angle = π * Σₗ [ Δₗ * ( x_{16*l+10} + η * (x_{16*l+10} - x_{16*l+11}) ) ]
         where Δₗ are non-uniform global weights and η is a fixed contrast parameter.
      
    Note: The input x is expected to have shape (80,).
      
    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      delta_weights (list): Global entanglement weights. Default: [0.15, 0.25, 0.35, 0.15, 0.10].
      eta (float): Contrast parameter for global fusion. Default: 0.1.
    """
    def __init__(self, n_qubits: int, delta_weights: list = None, eta: float = 0.1) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
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
            
            # Merged Entanglement Stage using ControlledPhaseShift gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                x_a = x[idx_a]
                x_b = x[idx_b]
                angle_cps = np.pi * (0.5 * x_a + 0.5 * x_b + 0.1 * (x_a - x_b))
                qml.ControlledPhaseShift(phi=angle_cps, wires=[j, (j + 1) % self.n_qubits])
            
            # Extended Entanglement using CRY gates for next-nearest neighbors
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                avg_val = (x[idx1] + x[idx2] + x[idx3]) / 3.0
                angle_cry = np.pi * avg_val
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])
        
        # Global Fusion: Aggregate inter-layer features via a MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * (x[base + 10] + self.eta * (x[base + 10] - x[base + 11]))
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
