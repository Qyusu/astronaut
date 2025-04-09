import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class OptimizedShallowEntanglementFeatureMapV12(BaseFeatureMap):
    """Optimized Shallow Entanglement Feature Map v12.
    
    This feature map distributes the 80-dimensional normalized input evenly over 10 qubits (each qubit i receives features f_{i,j} = x[8*i + j] for j = 0,...,7).
    Each qubit undergoes a local rotation block U_local^(i) which encodes features f₁–f₄ using a unique cyclic order selected from an expanded pool to maximize Bloch sphere coverage.
    A nearest-neighbor entanglement layer is implemented using a ring of ControlledPhaseShift (CP) gates.
    This is followed by a CRY layer connecting qubits with an offset of 2 (using the average of f₍i,4₎), an intermediate CRZ layer (offset of 3 using 2f₍i,5₎+3f₍(i+3),6₎ divided by 5), and a CRX layer (offset of 4 using the average of f₍i,2₎).
    The extra entanglement layer at offset 6 is removed to reduce circuit depth. Global correlations are integrated via a MultiRZ gate with rotation angle π*(global sum/110).
    Finally, a post-entanglement rotation block U_post^(i) applies sequential rotations RZ, RX, and RY based on features f₇ and f₈.
    """

    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Optimized Shallow Entanglement Feature Map v12.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Optimized Shallow Entanglement Feature Map v12.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,).
        """
        # Step 1: U_local - Local rotation block encoding features f1 to f4 with a cyclic permutation
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
        
        # Note: Extra entanglement (offset 6) is removed to reduce circuit depth
        
        # Step 6: Global entanglement using a MultiRZ gate
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
        global_angle = np.pi * (global_sum / 110.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 7: U_post - Post-entanglement rotations using features f7 and f8
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
