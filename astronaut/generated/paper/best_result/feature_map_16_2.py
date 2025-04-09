import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class HybridMultiAxisLocalEncodingWithISWAPEnhancedEntanglementFusionFeatureMap(BaseFeatureMap):
    """
    Hybrid Multi-Axis Local Encoding with ISWAP-Enhanced Entanglement Fusion Feature Map.

    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):

    - Local Encoding: The first 10 features are embedded onto a 10-qubit register using sequential rotations:
          RY(π * x[16*(l-1)+j]) followed by RX((π/2) * x[16*(l-1)+j]) for j = 0,...,9.

    - Stage 1 (Immediate Neighbor Entanglement):
          For each qubit j, compute from two designated entanglement features:
             a = x[16*(l-1) + 10 + (j mod 6)]
             b = x[16*(l-1) + 10 + ((j+1) mod 6)]
          Then compute:
             value = 0.5 * a + 0.5 * b + α * (a - b)
             angle = π * value
          Apply a controlled rotation between qubit j and (j+1) mod n_qubits using:
             - CRZ if the layer is odd
             - CRX if the layer is even

    - Stage 2 (Next-Nearest Neighbor Entanglement):
          For each qubit j, compute the triple average from features at indices (j mod 6), ((j+2) mod 6), and ((j+4) mod 6)
          from the block starting at 16*(l-1)+10. The rotation angle is given by π times this average,
          and a CRY gate is applied between qubit j and (j+2) mod n_qubits.

    - Stage 3 (ISWAP-Enhanced Entanglement):
          Apply a parameterized ISWAP-like interaction using the IsingXY gate between fixed qubit pairs.
          For j = 0,...,4, apply the gate on qubits j and j+5 with rotation angle π * γ.

    - Global Entanglement:
          A MultiRZ gate is applied across all qubits with rotation angle computed as:
             global_angle = π * Σₗ [ Δₗ * x[16*(l-1)+10] + κ * (x[16*(l-1)+10] - x[16*(l-1)+11]) ]
          where Δₗ (delta_weights) are adaptive weights and κ (kappa) is a contrast factor.

    Note: The input x is expected to have shape (80,).

    Parameters:
      n_qubits (int): Number of qubits (ideally 10).
      alpha (float): Contrast factor for Stage 1. Default is 0.3.
      delta_weights (list): A list of 5 global fusion weights. Default is [0.15, 0.25, 0.35, 0.15, 0.10].
      kappa (float): Contrast factor for global entanglement. Default is 0.3.
      gamma (float): Scaling factor for the ISWAP-enhanced entanglement. Default is 0.5.
    """

    def __init__(
        self, n_qubits: int, alpha: float = 0.3, delta_weights: list = None, kappa: float = 0.3, gamma: float = 0.5
    ) -> None:
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits

        # Contrast factor for Stage 1
        self.alpha = alpha

        # Global fusion weights
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights

        # Contrast factor for global entanglement
        self.kappa = kappa

        # Scaling factor for the ISWAP-enhanced entanglement (implemented via IsingXY gate)
        self.gamma = gamma

    def feature_map(self, x: np.ndarray) -> None:
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")

        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l

            # Local Encoding: Apply RY followed by RX rotations for qubits 0 through 9
            for j in range(self.n_qubits):
                angle_ry = np.pi * x[base + j]
                qml.RY(phi=angle_ry, wires=j)
                angle_rx = (np.pi / 2) * x[base + j]
                qml.RX(phi=angle_rx, wires=j)

            # Stage 1: Immediate Neighbor Entanglement
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                value = 0.5 * x[idx_a] + 0.5 * x[idx_b] + self.alpha * (x[idx_a] - x[idx_b])
                angle_ctrl = np.pi * value
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle_ctrl, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle_ctrl, wires=[j, (j + 1) % self.n_qubits])

            # Stage 2: Next-Nearest Neighbor Entanglement using CRY gates
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                avg_triple = (x[idx_a] + x[idx_b] + x[idx_c]) / 3.0
                angle_cry = np.pi * avg_triple
                qml.CRY(phi=angle_cry, wires=[j, (j + 2) % self.n_qubits])

            # Stage 3: ISWAP-Enhanced Entanglement using a parameterized IsingXY gate
            # Apply the gate between qubit pairs (0,5), (1,6), (2,7), (3,8), (4,9)
            for j in range(5):
                qml.IsingXY(phi=np.pi * self.gamma, wires=[j, j + 5])

        # Global Entanglement: Contrast Fusion via MultiRZ gate
        global_sum = 0.0
        for l in range(5):
            base = 16 * l
            global_sum += self.delta_weights[l] * x[base + 10] + self.kappa * (x[base + 10] - x[base + 11])
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
