import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class SimplifiedTripleStageFeatureMapWithUniformVarianceWeightsFeatureMap(BaseFeatureMap):
    """
    Simplified Triple-Stage Feature Map with Uniform Variance Weights.
    
    This feature map partitions the 80-dimensional input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - Local Encoding: The first 10 features are embedded onto a 10-qubit register via RY rotations:
            RY(π * x[16*(l-1) + j]) for j = 0,...,9.
      - Stage 1 (Immediate Neighbor Entanglement):
            For each qubit j, compute the rotation angle as
              θ₁ = 0.5 * x[16*(l-1) + 10 + (j mod 6)] + 0.5 * x[16*(l-1) + 10 + ((j+1) mod 6)],
            and apply a controlled rotation between qubit j and (j+1) mod n_qubits using:
              - CRZ if l is odd
              - CRX if l is even
      - Stage 2 (Next-Nearest Neighbor Entanglement):
            For each qubit j, compute the rotation angle as
              θ₂ = (1/3) * (x[16*(l-1) + 10 + (j mod 6)] + x[16*(l-1) + 10 + ((j+2) mod 6)] + x[16*(l-1) + 10 + ((j+4) mod 6)]),
            and apply a CRY gate between qubit j and (j+2) mod n_qubits with rotation angle π·θ₂.
      - Stage 3 (Mid-Range Entanglement):
            For each qubit j, compute the rotation angle as
              θ₃ = 0.5 * x[16*(l-1) + 10 + (j mod 6)] + 0.5 * x[16*(l-1) + 10 + ((j+1) mod 6)],
            scale it by a fixed factor λ = 0.4, and apply a ControlledPhaseShift gate between qubit j and (j+3) mod n_qubits with rotation angle π·λ·θ₃.
      - Global Entanglement: A MultiRZ gate is applied across all qubits with rotation angle
            π * (Σₗ (1/5) * x[16*(l-1) + 10]),
            aggregating the designated feature from each layer uniformly.
            (Here, 1/5 is used as the global weight.)
    
    Note: The input x is expected to have shape (80,).
    """
    def __init__(self, n_qubits: int, lambda_factor: float = 0.4, global_weight: float = 0.2) -> None:
        """
        Initialize the Simplified Triple-Stage Feature Map with Uniform Variance Weights.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            lambda_factor (float): Scaling factor for mid-range entanglement (default: 0.4).
            global_weight (float): Uniform global weight for the MultiRZ gate (default: 0.2, representing 1/5).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.lambda_factor = lambda_factor
        self.global_weight = global_weight
        
    def feature_map(self, x: np.ndarray) -> None:
        """
        Create the quantum circuit for the Simplified Triple-Stage Feature Map with Uniform Variance Weights.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers (l = 0,...,4 corresponds to layers 1,...,5)
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations for the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Stage 1: Immediate neighbor entanglement using CRZ/CRX gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                theta1 = 0.5 * x[idx_a] + 0.5 * x[idx_b]
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
                theta2 = (1/3) * (x[idx_a] + x[idx_b] + x[idx_c])
                angle2 = np.pi * theta2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage 3: Mid-range entanglement using ControlledPhaseShift gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                theta3 = 0.5 * x[idx_a] + 0.5 * x[idx_b]
                angle3 = np.pi * self.lambda_factor * theta3
                qml.ControlledPhaseShift(phi=angle3, wires=[j, (j + 3) % self.n_qubits])
        
        # Global entanglement: apply MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.global_weight * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
