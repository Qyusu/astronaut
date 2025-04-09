import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class GlobalMultiRZEnhancedLocalVariationFeatureMapV4(BaseFeatureMap):
    """Global MultiRZ Enhanced Local Variation Feature Map v4.
    
    This feature map partitions the 80-dimensional input into 10 blocks of 8 features each.
    For each qubit i (0 ≤ i ≤ 9), with features f_{i,j} = x[8*i + j] for j = 0,...,7:
      - U_primary: A primary local rotation block is applied with a cyclically varied order across qubits.
        For example, based on (i mod 3), the rotation order is one of:
          • [RX, RY, RZ, RX]
          • [RY, RZ, RX, RY]
          • [RZ, RX, RY, RZ]
        The rotations use features f1–f4 respectively (indices 0–3) scaled by π.
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates is applied between qubit i
        and qubit (i+1) mod 10 with a fixed angle (cp_angle).
      - Intermediate entanglement: CRZ gates are applied between qubit i and qubit (i+2) mod 10 with a rotation angle
        π * (3*f_{i,5} + f_{(i+2) mod 10,6})/4, where f_{i,5} is the 5th feature of qubit i (index 4) and f_{(i+2) mod 10,6}
        is the 6th feature (index 5) of the partner qubit.
      - Tertiary entanglement: CRX gates are applied between qubit i and qubit (i+4) mod 10 with a rotation angle
        π * (2*f_{i,5} - f_{(i+4) mod 10,6})/3.
      - Global entanglement: A MultiRZ gate is applied on all qubits with rotation angle set to π times the average of
        feature f_{i,7} (index 6) across all qubits.
      - U_post: A post-entanglement rotation block is applied locally on each qubit in the order RZ, RX, RY using
        angles π*f_{i,7}, π*f_{i,8}, and π*f_{i,7} respectively.
    The overall circuit implements
      |Φ(x)⟩ = (∏ U_post^(i)) · MultiRZ(π (∑_i f_{i,7}/10)) · (∏ CRX_{i,(i+4) mod 10}(π(2 f_{i,5} - f_{(i+4) mod 10,6})/3)) ·
              (∏ CRZ_{i,(i+2) mod 10}(π(3 f_{i,5} + f_{(i+2) mod 10,6})/4)) ·
              (∏ CP(q_i, q_{(i+1) mod 10})) · (∏ U_primary^(i)) |0⟩.
    """

    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Global MultiRZ Enhanced Local Variation Feature Map v4.
        
        Args:
            n_qubits (int): Number of qubits (should be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle

    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Global MultiRZ Enhanced Local Variation Feature Map v4.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,), with features normalized to [0, 1].
        """
        # Step 1: U_primary local rotations with cyclic variation in order
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Define cyclic orders based on qubit index modulo 3
            if qubit % 3 == 0:
                rotation_order = [qml.RX, qml.RY, qml.RZ, qml.RX]
            elif qubit % 3 == 1:
                rotation_order = [qml.RY, qml.RZ, qml.RX, qml.RY]
            else:
                rotation_order = [qml.RZ, qml.RX, qml.RY, qml.RZ]
            
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(rotation_order, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Intermediate entanglement using CRZ gates between qubit i and (i+2) mod n_qubits
        for qubit in range(self.n_qubits):
            partner = (qubit + 2) % self.n_qubits
            # f5 from current qubit (index 4) and f6 from partner (index 5)
            f5 = x[8 * qubit + 4]
            f6_partner = x[8 * partner + 5]
            angle_crz = np.pi * (3 * f5 + f6_partner) / 4.0
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 4: Tertiary entanglement using CRX gates between qubit i and (i+4) mod n_qubits
        for qubit in range(self.n_qubits):
            partner = (qubit + 4) % self.n_qubits
            f5 = x[8 * qubit + 4]
            f6_partner = x[8 * partner + 5]
            angle_crx = np.pi * (2 * f5 - f6_partner) / 3.0
            qml.CRX(phi=angle_crx, wires=[qubit, partner])
        
        # Step 5: Global entanglement using MultiRZ across all qubits
        # Compute the average of feature f7 (index 6) across all qubits
        total_f7 = 0.0
        for qubit in range(self.n_qubits):
            total_f7 += x[8 * qubit + 6]
        global_angle = np.pi * (total_f7 / self.n_qubits)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 6: U_post local rotations in the order RZ, RX, RY with features f7 and f8
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
