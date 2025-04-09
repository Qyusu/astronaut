import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class OptimizedShallowHybridEntanglementFeatureMapV17(BaseFeatureMap):
    """
    Optimized Shallow Hybrid Entanglement Feature Map v17.
    
    This feature map partitions an 80-dimensional normalized input among 10 qubits (with f_{i,j} = x[8*i + j]).
    Each qubit first executes a local rotation block U_local that encodes features f₁–f₄ using a unique cyclic permutation
    chosen from an expanded set, thereby diversifying state encoding across the Bloch sphere.
    The entanglement sequence is layered as follows:
      - A nearest-neighbor ControlledPhaseShift (CP) gate ring establishes immediate correlations.
      - A CRY layer (offset 2) connects qubits using the average of feature f₄.
      - A CRZ layer (offset 3) couples qubits based on π·(2f₅+3f₆_partner)/5.
      - A CRX layer (offset 4) uses π times the average of feature f₂.
      - An extra CRX layer (offset 6) applies a reduced rotation π·((f₃+f₃_partner)/6) to capture long‐range correlations.
    Global correlations are integrated via a MultiRZ gate with rotation angle π multiplied by the sum
    of selected features (f₁+f₂+f₃+f₅+f₆+2f₇+2f₈) divided by 100. Finally, a dedicated post-entanglement block U_post
    applies a fixed cyclic sequence of single-qubit rotations (RZ, RX, RY) based on features f₇ and f₈.
    
    U_local^(i) = R(π f₍i,1₎) R(π f₍i,2₎) R(π f₍i,3₎) R(π f₍i,4₎)
    U_post^(i)  = RZ(π f₍i,7₎) RX(π f₍i,8₎) RY(π f₍i,7₎)
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Optimized Shallow Hybrid Entanglement Feature Map v17.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Optimized Shallow Hybrid Entanglement Feature Map v17.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,).
        """
        # Step 1: U_local - Local rotation block encoding features f₁ to f₄ using a unique cyclic permutation
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Choose a cyclic permutation based on the qubit index
            if qubit % 3 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX]
            elif qubit % 3 == 1:
                rotations = [qml.RY, qml.RZ, qml.RX, qml.RY]
            else:
                rotations = [qml.RZ, qml.RX, qml.RY, qml.RZ]
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(rotations, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using a ControlledPhaseShift (CP) gate ring
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: CRY layer at offset 2 using the average of feature f₄
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: CRZ layer at offset 3 using features f₅ and f₆
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: CRX layer at offset 4 using the average of feature f₂
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Step 6: Extra CRX layer at offset 6 using feature f₃ for long-range correlations
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f3_current = x[base_idx + 2]
            partner = (qubit + 6) % self.n_qubits
            f3_partner = x[8 * partner + 2]
            extra_crx_angle = np.pi * ((f3_current + f3_partner) / 6.0)
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
        
        # Step 8: U_post block - Post-entanglement rotations using features f₇ and f₈
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
