import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class AugmentedGlobalOffsetRichEntanglementFeatureMapV9(BaseFeatureMap):
    """Augmented Global and Offset-Rich Entanglement Feature Map v9.
    
    This feature map distributes an 80-dimensional normalized input evenly over 10 qubits (each qubit i receives features f_{i,j} = x[8*i + j] for j = 0,...,7).
    The circuit applies the following operations:
      - U_local: A local rotation block U_local^(i) is applied on each qubit, encoding features f₁–f₄ using a unique cyclic permutation.
                  For example, even-indexed qubits use RX, RY, RZ, RX while odd-indexed qubits use RY, RX, RZ, RY.
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates is applied between qubit i and qubit (i+1) mod 10.
      - Additional entanglement: A CRY gate layer with an offset of 2 is applied, with rotation angle π*((f₄_current + f₄_partner)/2).
      - Intermediate entanglement: CRZ gates connect qubit i with qubit (i+3) mod 10, with rotation angle π*(2·f₅ + 3·f₆_partner)/5.
      - Novel entanglement: A CRX layer connects qubit i with qubit (i+4) mod 10, using rotation angle π*((f₂_current + f₂_partner)/2).
      - Extended entanglement: An additional CRY layer connects qubit i with qubit (i+5) mod 10, with rotation angle π*((f₃_current + f₃_partner)/2).
      - Global entanglement: A MultiRZ gate is applied across all qubits with rotation angle π*(Σᵢ(f₁+f₂+f₅+f₆+2f₇+2f₈))/80.
      - U_post: A post-entanglement rotation block U_post^(i) applies RZ and RX rotations using features f₇ and f₈ respectively.
    
    The overall operation is given by:
      |Φ(x)⟩ = (∏ U_post^(i)) · MultiRZ(π*(Σᵢ(f₁+f₂+f₅+f₆+2f₇+2f₈))/80) · (∏ CRY_{i,(i+5) mod 10}(π*((f₃+f₃_partner)/2))) ·
               (∏ CRX_{i,(i+4) mod 10}(π*((f₂+f₂_partner)/2))) · (∏ CRZ_{i,(i+3) mod 10}(π*(2f₅+3f₆_partner)/5)) ·
               (∏ CRY_{i,(i+2) mod 10}(π*((f₄+f₄_partner)/2))) · (∏ CP(q_i, q_{(i+1) mod 10})) · (∏ U_local^(i)) |0⟩,
    where
      U_local^(i) = R_{α_{i,1}}(π·f₁) R_{α_{i,2}}(π·f₂) R_{α_{i,3}}(π·f₃) R_{α_{i,4}}(π·f₄),
      U_post^(i)  = RZ(π·f₇) RX(π·f₈), with f_{i,j} = x[8*i + j].
    """

    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Augmented Global and Offset-Rich Entanglement Feature Map v9.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Augmented Global and Offset-Rich Entanglement Feature Map v9.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,).
        """
        # Step 1: U_local - Local rotation block encoding features f1 to f4
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Select a unique cyclic permutation based on qubit index
            if qubit % 2 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX]
            else:
                rotations = [qml.RY, qml.RX, qml.RZ, qml.RY]
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(rotations, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Additional entanglement using CRY gates (offset of 2)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: Intermediate entanglement using CRZ gates (offset of 3)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: Novel entanglement using CRX gates (offset of 4)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Step 6: Extended entanglement using CRY gates (offset of 5)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f3_current = x[base_idx + 2]
            partner = (qubit + 5) % self.n_qubits
            f3_partner = x[8 * partner + 2]
            cry_angle = np.pi * ((f3_current + f3_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 7: Global entanglement using a MultiRZ gate
        global_sum = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f5 = x[base_idx + 4]
            f6 = x[base_idx + 5]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            global_sum += (f1 + f2 + f5 + f6 + 2 * f7 + 2 * f8)
        global_angle = np.pi * (global_sum / 80.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 8: U_post - Post-entanglement rotations using features f7 and f8
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
