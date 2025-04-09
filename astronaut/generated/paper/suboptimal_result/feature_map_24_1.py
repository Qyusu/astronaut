import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class AdaptiveHierarchicalEntanglementFeatureMapV24(BaseFeatureMap):
    """
    Adaptive Hierarchical Entanglement Feature Map v24.
    
    This feature map partitions an 80-dimensional normalized input among 10 qubits (with f(i,j) = x[8*i + j]).
    Each qubit applies a local rotation block U_local^(i) to encode features f₁–f₄ using variable cyclic orders chosen from an expanded set,
    ensuring full Bloch sphere coverage. Qubits are subdivided into two blocks (Block A: qubits 0–4, Block B: qubits 5–9), where intra‐block
    entanglement is achieved via a dense ControlledPhaseShift (CP) gate network. An inter‐block CP gate (applied between qubits 2 and 7,
    corresponding to q₃ and q₈ in one-indexed notation) further couples the blocks. Standard offset entanglement layers follow:
      - A CRY layer at offset 2 using π*((f₄_current + f₄_partner)/2).
      - A CRZ layer at offset 3 using π*(2f₅ + 3f₆_partner)/5.
      - A CRX layer at offset 4 using π*((f₂_current + f₂_partner)/2).
      - An extra CRX layer at offset 6 using π*((f₃_current + f₃_partner)/6).
    Global correlations are fused via a MultiRZ gate with rotation angle π*(Σ(f₁+f₂+f₃+f₅+f₆+2f₇+2f₈)/97).
    Finally, a post-entanglement block U_post^(i) applies a fixed cyclic sequence: RZ(π·f₇) → RX(π·f₈) → RY(π·f₇).
    Hardware-aware error mitigation is assumed to be applied at the device level.
    """
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4, global_const: float = 97) -> None:
        """Initialize the Adaptive Hierarchical Entanglement Feature Map v24.
        
        Args:
            n_qubits (int): Number of qubits (should be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for ControlledPhaseShift gates (default: π/4).
            global_const (float): Normalization constant for the MultiRZ gate (default: 97).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        self.global_const = global_const

    def feature_map(self, x: np.ndarray) -> None:
        """Construct the quantum circuit for the Adaptive Hierarchical Entanglement Feature Map v24.
        
        Args:
            x (np.ndarray): Input normalized feature vector of shape (80,).
        """
        # Step 1: U_local block - encode features f₁ to f₄ with variable cyclic orders
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Choose cyclic rotation order from an expanded pool based on qubit index
            if qubit % 3 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX]
            elif qubit % 3 == 1:
                rotations = [qml.RY, qml.RZ, qml.RX, qml.RY]
            else:
                rotations = [qml.RZ, qml.RX, qml.RY, qml.RZ]
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(rotations, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Hierarchical entanglement
        # Intra-block entanglement for Block A (qubits 0 to 4)
        for j in range(0, 5):
            for k in range(j + 1, 5):
                qml.ControlledPhaseShift(phi=self.cp_angle, wires=[j, k])
        
        # Intra-block entanglement for Block B (qubits 5 to 9)
        for j in range(5, self.n_qubits):
            for k in range(j + 1, self.n_qubits):
                qml.ControlledPhaseShift(phi=self.cp_angle, wires=[j, k])
        
        # Inter-block entanglement: apply CP between qubit 2 and qubit 7 (corresponding to q₃ and q₈ in one-indexed notation)
        qml.ControlledPhaseShift(phi=self.cp_angle, wires=[2, 7])
        
        # Step 3: Standard offset entanglement layers
        # Offset-2 CRY layer (using feature f₄)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Offset-3 CRZ layer (using features f₅ and f₆)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Offset-4 CRX layer (using feature f₂)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Offset-6 Extra CRX layer (using feature f₃)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f3_current = x[base_idx + 2]
            partner = (qubit + 6) % self.n_qubits
            f3_partner = x[8 * partner + 2]
            extra_crx_angle = np.pi * ((f3_current + f3_partner) / 6.0)
            qml.CRX(phi=extra_crx_angle, wires=[qubit, partner])
        
        # Step 4: Global entanglement using MultiRZ
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
        global_angle = np.pi * (global_sum / self.global_const)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 5: U_post block - apply fixed cyclic rotations on features f₇ and f₈
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
