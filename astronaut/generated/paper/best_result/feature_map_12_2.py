import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class DualStageWithIntegratedZeroNoiseExtrapolationAndUnifiedGlobalFusionFeatureMap(BaseFeatureMap):
    """
    Dual-Stage with Integrated Zero-Noise Extrapolation and Unified Global Fusion Feature Map.
    
    This feature map divides the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 0,...,4):
      - Local Encoding: The first 10 features are embedded onto a 10-qubit register via RY rotations:
            RY(π * x[16*l + j]) for j = 0,...,9.
      - Stage A (Immediate Neighbor Entanglement):
            For each qubit j, compute the rotation angle as
              ψ₁ = w_prime[l][j] * x[16*l + 10 + (j mod 6)] + (1 - w_prime[l][j]) * x[16*l + 10 + ((j+1) mod 6)],
            and apply a controlled rotation between qubit j and (j+1) mod n_qubits using:
              - CRZ if (l+1) is odd
              - CRX if (l+1) is even
      - Stage B (Next-Nearest Neighbor Entanglement):
            For each qubit j, compute the rotation angle as
              ψ₂ = v_prime[l][j][0] * x[16*l + 10 + (j mod 6)] +
                   v_prime[l][j][1] * x[16*l + 10 + ((j+2) mod 6)] +
                   v_prime[l][j][2] * x[16*l + 10 + ((j+4) mod 6)],
            and apply a CRY gate between qubit j and (j+2) mod n_qubits with rotation angle π·ψ₂.
      - Global Entanglement: A MultiRZ gate is applied across all qubits with rotation angle
            π * (Σₗ δ′ₗ * x[16*l + 10]),
            where δ′ₗ are non-uniform weights determined offline.
    
    Integrated zero-noise extrapolation (along with measurement twirling and bit-flip tolerance) is assumed to be incorporated offline to enhance robustness against hardware noise.
    
    Note: The input x is expected to have shape (80,).
    """
    def __init__(self, n_qubits: int, w_prime: list = None, v_prime: list = None, delta_weights_prime: list = None, zero_noise_extrapolation: bool = True) -> None:
        """
        Initialize the Dual-Stage with Integrated Zero-Noise Extrapolation and Unified Global Fusion Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            w_prime (list): 5x10 weight matrix for Stage A (default: all 0.5).
            v_prime (list): 5x10x3 weight tensor for Stage B (default: all 1/3).
            delta_weights_prime (list): List of 5 weights for global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
            zero_noise_extrapolation (bool): Flag to indicate integration of zero-noise extrapolation (default: True).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        
        if w_prime is None:
            self.w_prime = [[0.5 for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.w_prime = w_prime
        
        if v_prime is None:
            self.v_prime = [[[1/3 for _ in range(3)] for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.v_prime = v_prime
        
        if delta_weights_prime is None:
            self.delta_weights_prime = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights_prime = delta_weights_prime
        
        self.zero_noise_extrapolation = zero_noise_extrapolation
        
    def feature_map(self, x: np.ndarray) -> None:
        """
        Create the quantum circuit for the Dual-Stage with Integrated Zero-Noise Extrapolation and Unified Global Fusion Feature Map.
        
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
            
            # Stage A: Immediate neighbor entanglement
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                psi1 = self.w_prime[l][j] * x[idx_a] + (1 - self.w_prime[l][j]) * x[idx_b]
                angle1 = np.pi * psi1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage B: Next-nearest neighbor entanglement
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                psi2 = (self.v_prime[l][j][0] * x[idx_a] +
                        self.v_prime[l][j][1] * x[idx_b] +
                        self.v_prime[l][j][2] * x[idx_c])
                angle2 = np.pi * psi2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
        
        # Global entanglement: apply MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights_prime[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
