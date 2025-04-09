import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class UnifiedMultiPermutationExpandedGlobalFeatureMapV13(BaseFeatureMap):
    """Unified Multi-Permutation Expanded Global Feature Map v13.
    
    This feature map evenly allocates an 80-dimensional normalized input to 10 qubits, with each qubit i receiving features f_{i,1} through f_{i,8} (with f_{i,j} = x[8*i + j]).
    A unified rotation block U_unified^(i) encodes features f₁, f₂, f₃, f₄, f₇, and f₈ concurrently using a fixed cyclic permutation chosen from an expanded set to enhance state diversity.
    The entanglement layers consist of a nearest-neighbor CP gate ring; a CRY layer (offset 2) based on the average of f₄; a CRZ layer (offset 3) using 2f₅+3f₆ divided by 5; a CRX layer (offset 4) using the average of f₂; and an extra CRX layer (offset 6) with a toned-down rotation angle based on f₃.
    Global correlations are integrated via a MultiRZ gate with rotation angle computed as π times the weighted sum of f₁, f₂, f₃, f₅, f₆, 2f₇, and 2f₈ divided by 105.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Unified Multi-Permutation Expanded Global Feature Map v13.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Unified Multi-Permutation Expanded Global Feature Map v13.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,).
        """
        # Step 1: U_unified - Unified rotation block encoding features f1, f2, f3, f4, f7, and f8
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            
            # Select a fixed cyclic permutation from an expanded set based on qubit index modulo 3
            if qubit % 3 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX, qml.RZ, qml.RY]
            elif qubit % 3 == 1:
                rotations = [qml.RY, qml.RZ, qml.RX, qml.RY, qml.RX, qml.RZ]
            else:
                rotations = [qml.RZ, qml.RX, qml.RY, qml.RZ, qml.RY, qml.RX]
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4, np.pi * f7, np.pi * f8]
            for gate, angle in zip(rotations, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using a CP gate
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Additional entanglement using a CRY gate (offset 2) with the average of f4
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: Intermediate entanglement using a CRZ gate (offset 3) with f5 and f6
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: Novel entanglement using a CRX gate (offset 4) with the average of f2
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Step 6: Extra entanglement using a CRX gate (offset 6) with a toned-down rotation angle using f3
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
        global_angle = np.pi * (global_sum / 105.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
