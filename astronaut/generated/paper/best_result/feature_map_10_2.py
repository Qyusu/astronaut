import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class HybridDualStageMergedCrossLayerEntanglementFeatureMap(BaseFeatureMap):
    """
    Hybrid Dual-Stage with Merged Cross-Layer Entanglement and Uniform Weighting Feature Map.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 0,...,4):
      - Local encoding: The first 10 features are encoded on a 10-qubit register via RY rotations:
            RY(π * x[16*l + j]) for j = 0,...,9.
      - Stage 1 (Immediate neighbor entanglement):
            For each qubit j, a controlled rotation is applied between qubit j and its cyclic neighbor ((j+1) mod 10).
            The rotation angle is computed as π times the uniform average of two designated features:
                θ₁ = (x[16*l + 10 + (j mod 6)] + x[16*l + 10 + ((j+1) mod 6)])/2.
            CRZ is used for odd layers (l+1 odd) and CRX for even layers.
      - Stage 2 (Next-nearest neighbor entanglement):
            For each qubit j, a CRY gate entangles qubit j with qubit ((j+2) mod 10) using an angle computed as
                θ₂ = (x[16*l + 10 + (j mod 6)] + x[16*l + 10 + ((j+1) mod 6)] + x[16*l + 10 + ((j+2) mod 6)])/3.
      - Global entanglement: A MultiRZ gate is applied the combined state with a rotation angle
                π * (∑ₗ (x[16*l + 10] / 5)).
    
    This design merges cross-layer entanglement into the final global MultiRZ gate, thereby reducing circuit depth
    while preserving uniform, linear encoding of inter-layer correlations.
    
    Note: The input x is expected to have shape (80,).
    """
    def __init__(self, n_qubits: int, merged_cross: bool = True) -> None:
        """
        Initialize the Hybrid Dual-Stage with Merged Cross-Layer Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            merged_cross (bool): Flag indicating whether cross-layer entanglement is merged into the global entanglement stage (default: True).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.merged_cross = merged_cross
        
    def feature_map(self, x: np.ndarray) -> None:
        """
        Create the quantum circuit for the Hybrid Dual-Stage with Merged Cross-Layer Entanglement Feature Map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
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
                avg_val = (x[idx_a] + x[idx_b]) / 2.0
                angle_stage1 = np.pi * avg_val
                if l % 2 == 0:
                    qml.CRZ(phi=angle_stage1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle_stage1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                idx_c = base + 10 + ((j + 2) % 6)
                avg_val = (x[idx_a] + x[idx_b] + x[idx_c]) / 3.0
                angle_stage2 = np.pi * avg_val
                qml.CRY(phi=angle_stage2, wires=[j, (j + 2) % self.n_qubits])
        
        # Global entanglement: apply MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += x[16 * l + 10] / 5.0
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
