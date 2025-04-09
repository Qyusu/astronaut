import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class EnhancedGlobalWeightedMultiRZPreEntanglementHybridFeatureMapV5(BaseFeatureMap):
    """Enhanced Global Weighted MultiRZ Pre-Entanglement Hybrid Feature Map v5.
    
    This feature map partitions the 80-dimensional input into 10 blocks of 8 features each.
    For each qubit i (0 ≤ i ≤ 9), with features f_{i,j} = x[8*i + j] (for j = 0,...,7):
      - U_pre: A pre-entanglement rotation block applied using features f_{i,1}–f_{i,4} (indices 0–3) with a
                cyclic order determined by the qubit index. For example, even-indexed qubits use [RX, RZ, RY, RX],
                while odd-indexed qubits use [RY, RX, RZ, RY].
      - U_primary: A primary local rotation block applied in a fixed order [RX, RY, RZ, RX] using the same
                   features f_{i,1}–f_{i,4}.
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates is applied between qubit i and
                                        qubit (i+1) mod 10.
      - Intermediate entanglement: CRZ gates are applied between qubit i and qubit (i+3) mod 10 with rotation
                                    angles π*(3*f_{i,5} - f_{(i+3),6})/4, where f_{i,5} is from index 4 of qubit i
                                    and f_{(i+3),6} is from index 5 of the partner qubit.
      - Maximal-separation entanglement: CRY gates are applied between qubit i and qubit (i+5) mod 10 with rotation
                                          angles π*(f_{i,8}), using feature f_{i,8} (index 7) of the current qubit.
      - Global entanglement: A MultiRZ gate applied on all qubits with rotation angle calculated as
                               π * (sum over qubits of (f5 + 2*f6 + f7 + 2*f8))/(6*n_qubits).
      - U_post: A post-entanglement rotation block applied in the order RX, RZ, RY using features f_{i,7} and
                f_{i,8} as follows: RX(π*f_{i,7}), then RZ(π*f_{i,8}), and finally RY(π*((f_{i,7}+f_{i,8})/2)).
    
    The overall circuit implements:
      |Φ(x)⟩ = U_post · MultiRZ(π*(Σ(f5+2f6+f7+2f8)/(6*n_qubits))) · (∏ CRY_{i,(i+5) mod 10}(π*f_{i,8})) ·
              (∏ CRZ_{i,(i+3) mod 10}(π*(3*f5 - f6_partner)/4)) · (∏ CP(q_i, q_{(i+1) mod 10})) ·
              U_primary · U_pre |0⟩.

    This design maintains full data fidelity while leveraging both local and global entanglement to enhance
    the quantum feature encoding. Error mitigation strategies can be incorporated externally if required.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Enhanced Global Weighted MultiRZ Pre-Entanglement Hybrid Feature Map v5.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Enhanced Global Weighted MultiRZ Pre-Entanglement Hybrid Feature Map v5.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,), with features normalized to [0, 1].
        """
        # Step 1: U_pre - Pre-entanglement rotation block with cyclic order based on qubit index
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
        
        # Step 4: Intermediate entanglement using CRZ gates between qubit i and (i+3) mod n_qubits
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            angle_crz = np.pi * (3 * f5 - f6_partner) / 4.0
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 5: Maximal-separation entanglement using CRY gates between qubit i and (i+5) mod n_qubits
        for qubit in range(self.n_qubits):
            # Use the current qubit's f8 (feature index 7) for the rotation angle
            base_idx = 8 * qubit
            f8 = x[base_idx + 7]
            partner = (qubit + 5) % self.n_qubits
            angle_cry = np.pi * f8
            qml.CRY(phi=angle_cry, wires=[qubit, partner])
        
        # Step 6: Global entanglement using MultiRZ across all qubits
        total = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            f6 = x[base_idx + 5]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            total += (f5 + 2 * f6 + f7 + 2 * f8)
        global_angle = np.pi * (total / (6.0 * self.n_qubits))
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 7: U_post - Post-entanglement rotation block in order: RX, RZ, RY
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RX(phi=np.pi * f7, wires=[qubit])
            qml.RZ(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * ((f7 + f8) / 2), wires=[qubit])
        
        # Note: Optional post-processing error mitigation (e.g., Bit Flip Tolerance) can be applied externally.
