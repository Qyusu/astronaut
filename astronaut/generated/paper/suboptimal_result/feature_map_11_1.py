import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class ExtraOffsetEnhancedMultiEntanglementFeatureMapV11(BaseFeatureMap):
    """Extra-Offset Enhanced Multi-Entanglement Feature Map v11.
    
    This feature map distributes an 80-dimensional normalized input evenly over 10 qubits (each qubit i receives features f_{i,j} = x[8*i + j] for j = 0,...,7).
    The circuit applies the following operations:
      - U_local: A local rotation block U_local^(i) is applied on each qubit, encoding features f₁–f₄ in a variable cyclic order selected from an expanded pool.
                For example, the rotation order is chosen based on (qubit mod 3):
                  * 0: RX, RY, RZ, RX
                  * 1: RY, RZ, RX, RY
                  * 2: RZ, RX, RY, RZ
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates connects qubit i with qubit (i+1) mod 10.
      - Additional entanglement 1: A CRY layer with an offset of 2 is applied, with rotation angle π·((f₍i,4₎ + f₍(i+2),4₎)/2).
      - Intermediate entanglement: CRZ gates connect qubit i with qubit (i+3) mod 10, with rotation angle π·(2·f₍i,5₎ + 3·f₍(i+3),6₎)/5.
      - Novel entanglement: A CRX layer connects qubit i with qubit (i+4) mod 10, using rotation angle π·((f₍i,2₎ + f₍(i+4),2₎)/2).
      - Extra entanglement: A new CRX layer connects qubit i with qubit (i+6) mod 10, using rotation angle π·((f₍i,3₎ + f₍(i+6),3₎)/2).
      - Global entanglement: A MultiRZ gate is applied across all qubits with rotation angle π·(Σᵢ(f₍i,1₎ + f₍i,2₎ + f₍i,3₎ + f₍i,5₎ + f₍i,6₎ + 2·f₍i,7₎ + 2·f₍i,8₎))/100.
      - U_post: A post-entanglement rotation block U_post^(i) applies sequential rotations RZ, RX, and RY using features f₇ and f₈, with RZ and RY both using f₇.
    
    The overall operation is given by:
      |Φ(x)⟩ = (∏ U_post^(i)) · MultiRZ(π·(Σᵢ(f₁+f₂+f₃+f₅+f₆+2f₇+2f₈))/100) · (∏ CRX_{i,(i+6) mod 10}(π·((f₃+f₍i+6₎,3)/2))) ·
               (∏ CRX_{i,(i+4) mod 10}(π·((f₂+f₍i+4₎,2)/2))) · (∏ CRZ_{i,(i+3) mod 10}(π·(2f₍i,5₎+3f₍(i+3),6₎)/5)) ·
               (∏ CRY_{i,(i+2) mod 10}(π·((f₍i,4₎+f₍(i+2),4₎)/2))) · (∏ CP(q_i, q_{(i+1) mod 10})) · (∏ U_local^(i)) |0⟩,
    where
      U_local^(i) = R_{α_{i,1}}(π·f₍i,1₎) R_{α_{i,2}}(π·f₍i,2₎) R_{α_{i,3}}(π·f₍i,3₎) R_{α_{i,4}}(π·f₍i,4₎),
      U_post^(i)  = RZ(π·f₍i,7₎) RX(π·f₍i,8₎) RY(π·f₍i,7₎), with f₍i,j₎ = x[8*i + j].
    """

    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Extra-Offset Enhanced Multi-Entanglement Feature Map v11.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Extra-Offset Enhanced Multi-Entanglement Feature Map v11.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,).
        """
        # Step 1: U_local - Local rotation block encoding features f1 to f4 using an expanded cyclic permutation
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Select a cyclic permutation based on qubit index modulo 3
            if qubit % 3 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX]
            elif qubit % 3 == 1:
                rotations = [qml.RY, qml.RZ, qml.RX, qml.RY]
            else:
                rotations = [qml.RZ, qml.RX, qml.RY, qml.RZ]
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(rotations, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Additional entanglement using CRY gates (offset of 2) with f4
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: Intermediate entanglement using CRZ gates (offset of 3) with f5 and f6
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: Novel entanglement using CRX gates (offset of 4) with f2
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Step 6: Extra entanglement using an additional CRX gate (offset of 6) with f3
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f3_current = x[base_idx + 2]
            partner = (qubit + 6) % self.n_qubits
            f3_partner = x[8 * partner + 2]
            extra_crx_angle = np.pi * ((f3_current + f3_partner) / 2.0)
            qml.CRX(phi=extra_crx_angle, wires=[qubit, partner])
        
        # Step 7: Global entanglement using a MultiRZ gate
        global_sum = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f5 = x[base_idx + 4]
            f6 = x[base_idx + 5]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            global_sum += (f1 + f2 + f3 + f5 + f6 + 2 * f7 + 2 * f8)
        global_angle = np.pi * (global_sum / 100.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 8: U_post - Post-entanglement rotations using features f7 and f8
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
