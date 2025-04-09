import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class MaximalSeparationPreEntanglementAugmentedFeatureMapV5(BaseFeatureMap):
    """Maximal-Separation & Pre-Entanglement Augmented Feature Map v5.

    This feature map partitions the 80-dimensional input into 10 blocks of 8 features each.
    For each qubit i (0 ≤ i ≤ 9), with features f_{i,j} = x[8*i + j] (j = 0,...,7):
      - U_pre: A pre-entanglement rotation block applied using features f_{i,1}–f_{i,4} (indices 0–3).
                The cyclic order is varied by qubit index: even-indexed qubits use [RX, RZ, RY, RX],
                whereas odd-indexed qubits use [RY, RX, RZ, RY].
      - U_primary: A primary local rotation block applied in a fixed order [RX, RY, RZ, RX] using
                   the same features f_{i,1}–f_{i,4}.
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates between qubit i and
                                        qubit (i+1) mod 10 with a fixed phase angle.
      - Intermediate entanglement: CRZ gates between qubit i and qubit (i+2) mod 10 with rotation angles
                                    π*(2*f_{i,5} + f_{(i+2),6})/3, where f_{i,5} is feature index 4 of qubit i
                                    and f_{(i+2),6} is feature index 5 of the partner qubit.
      - Maximal-separation entanglement: CRX gates between qubit i and qubit (i+5) mod 10 with rotation angles
                                          π*(f_{i,7} - f_{(i+5),8}), where f_{i,7} is feature index 6 of qubit i
                                          and f_{(i+5),8} is feature index 7 of the partner qubit.
      - Global entanglement: A MultiRZ gate acting on all qubits with rotation angle computed as
                               π * (sum over qubits of (2*f5 + 2*f6 + f7 + f8))/(6*n_qubits).
      - U_post: A post-entanglement rotation block applied in the order RZ, RX, RY using features f_{i,7}
                and f_{i,8} as follows: RZ(π*f_{i,7}), RX(π*f_{i,8}), RY(π*f_{i,7}).

    The overall circuit implements:
      |Φ(x)⟩ = U_post · MultiRZ(π*(Σ(2f5+2f6+f7+f8)/(6*n_qubits))) · (∏ CRX_{i,(i+5) mod 10}(π*(f7 - f8_partner))) ·
              (∏ CRZ_{i,(i+2) mod 10}(π*(2f5+f6_partner)/3)) · (∏ CP(q_i, q_{(i+1) mod 10})) ·
              U_primary · U_pre |0⟩.

    An optional error mitigation strategy (e.g., Bit Flip Tolerance) may be applied during post-processing.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Maximal-Separation & Pre-Entanglement Augmented Feature Map v5.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Maximal-Separation & Pre-Entanglement Augmented Feature Map v5.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,), with features normalized to [0, 1].
        """
        # Step 1: U_pre - Pre-entanglement rotation block with varied cyclic order
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            if qubit % 2 == 0:
                pre_order = [qml.RX, qml.RZ, qml.RY, qml.RX]
            else:
                pre_order = [qml.RY, qml.RX, qml.RZ, qml.RY]
            
            angles_pre = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(pre_order, angles_pre):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: U_primary - Primary local rotation block in fixed order: RX, RY, RZ, RX
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            qml.RX(phi=np.pi * f1, wires=[qubit])
            qml.RY(phi=np.pi * f2, wires=[qubit])
            qml.RZ(phi=np.pi * f3, wires=[qubit])
            qml.RX(phi=np.pi * f4, wires=[qubit])
        
        # Step 3: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 4: Intermediate entanglement using CRZ gates between qubit i and (i+2) mod n_qubits
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 2) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            angle_crz = np.pi * (2 * f5 + f6_partner) / 3.0
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 5: Maximal-separation entanglement using CRX gates between qubit i and (i+5) mod n_qubits
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            partner = (qubit + 5) % self.n_qubits
            f8_partner = x[8 * partner + 7]
            angle_crx = np.pi * (f7 - f8_partner)
            qml.CRX(phi=angle_crx, wires=[qubit, partner])
        
        # Step 6: Global entanglement using MultiRZ across all qubits
        total = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            f6 = x[base_idx + 5]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            total += (2 * f5 + 2 * f6 + f7 + f8)
        global_angle = np.pi * (total / (6.0 * self.n_qubits))
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 7: U_post - Post-entanglement rotation block in order: RZ, RX, RY
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
        
        # Note: An optional error mitigation step (e.g., Bit Flip Tolerance) can be applied externally.
