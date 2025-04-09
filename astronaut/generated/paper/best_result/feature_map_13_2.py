import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DualStageWithIntegratedMidRangeViaCRotAndZeroNoiseExtrapolationFeatureMap(BaseFeatureMap):
    """
    Dual-Stage with Integrated Mid-Range via CRot and Zero-Noise Extrapolation Feature Map.
    
    This feature map divides the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - Local Encoding: The first 10 features are embedded onto a 10-qubit register via RY rotations:
            RY(π * x[16*(l-1) + j]) for j = 0,...,9.
      - Stage A (Immediate Neighbor Entanglement):
            For each qubit j, compute the angle as
              ψ₁ = 0.5 * x[16*(l-1) + 10 + (j mod 6)] + 0.5 * x[16*(l-1) + 10 + ((j+1) mod 6)],
            and apply a controlled rotation between qubit j and (j+1) mod n_qubits using:
              - CRZ if l is odd
              - CRX if l is even
      - Stage B (Next-Nearest Neighbor Entanglement):
            For each qubit j, compute the angle as
              ψ₂ = (1/3) * (x[16*(l-1) + 10 + (j mod 6)] + x[16*(l-1) + 10 + ((j+2) mod 6)] + x[16*(l-1) + 10 + ((j+4) mod 6)]),
            and apply a CRY gate between qubit j and (j+2) mod n_qubits with rotation angle π·ψ₂.
      - Stage C (Mid-Range Entanglement via CRot):
            For each qubit j, compute the angle as
              ψ₃ = 0.5 * x[16*(l-1) + 10 + (j mod 6)] + 0.5 * x[16*(l-1) + 10 + ((j+1) mod 6)],
            scale it by a factor of 0.4, and apply a CRot gate between qubit j and (j+3) mod n_qubits with rotation parameters set to effect a rotation of π·0.4·ψ₃.
      - Global Entanglement: A MultiRZ gate is applied across all qubits with rotation angle
            π * (Σₗ δ′ₗ * x[16*(l-1) + 10]),
            where the δ′ₗ are non-uniform weights (default: [0.15, 0.25, 0.35, 0.15, 0.10]) derived offline.
    
    Advanced error mitigation techniques such as zero-noise extrapolation (enabled by the flag zero_noise_extrapolation),
    measurement twirling, and improved circuit scheduling are assumed to be integrated offline.
    
    Note: The input x is expected to have shape (80,).
    """
    def __init__(self, n_qubits: int, delta_weights_prime: list = None, mid_range_scale: float = 0.4, zero_noise_extrapolation: bool = True) -> None:
        """
        Initialize the Dual-Stage with Integrated Mid-Range via CRot and Zero-Noise Extrapolation Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            delta_weights_prime (list): List of 5 weights for global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
            mid_range_scale (float): Scaling factor for the mid-range CRot stage (default: 0.4).
            zero_noise_extrapolation (bool): Flag indicating integration of zero-noise extrapolation (default: True).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        if delta_weights_prime is None:
            self.delta_weights_prime = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights_prime = delta_weights_prime
        self.mid_range_scale = mid_range_scale
        self.zero_noise_extrapolation = zero_noise_extrapolation
        
    def feature_map(self, x: np.ndarray) -> None:
        """
        Create the quantum circuit for the Dual-Stage with Integrated Mid-Range via CRot and Zero-Noise Extrapolation Feature Map.
        
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
            
            # Stage A: Immediate neighbor entanglement using CRZ/CRX gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                psi1 = 0.5 * x[idx_a] + 0.5 * x[idx_b]
                angle1 = np.pi * psi1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage B: Next-nearest neighbor entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                psi2 = (1/3) * (x[idx_a] + x[idx_b] + x[idx_c])
                angle2 = np.pi * psi2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage C: Mid-range entanglement using CRot gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                psi3 = 0.5 * x[idx_a] + 0.5 * x[idx_b]
                angle3 = np.pi * self.mid_range_scale * psi3
                # Use CRot with rotation parameters: here we set theta and omega to 0 to focus the rotation in one parameter
                qml.CRot(phi=angle3, theta=0.0, omega=0.0, wires=[j, (j + 3) % self.n_qubits])
        
        # Global entanglement: apply MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights_prime[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
