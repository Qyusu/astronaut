import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class AdaptiveFullyConnectedGlobalFeatureMapV26(BaseFeatureMap):
    """
    Adaptive Fully Connected Global Feature Map v26.
    
    This feature map partitions an 80-dimensional normalized input among 10 qubits (with f(i,j) = x[8*i + j]).
    Each qubit first applies a local rotation block U_local^(i) that encodes features f₁–f₄ using variable cyclic
    permutations selected from an expanded set to maximize coverage of the Bloch sphere. The circuit then employs a layered
    entanglement strategy:
      - A fully connected CP gate ring entangles adjacent qubits,
      - A CRY layer at offset 2 applies rotations based on π*((f(i,4) + f((i+2),4))/2),
      - A CRZ layer at offset 3 applies rotations π*(2f(i,5) + 3f((i+3),6))/5,
      - A CRX layer at offset 4 uses rotations π*((f(i,2) + f((i+4),2))/2),
      - An extra CRX layer at offset 6 uses rotations π*((f(i,3) + f((i+6),3))/6),
      - An ultra‐light CRY layer at offset 5 applies rotations scaled by 1/32, i.e. π*((f(i,8) + f((i+5),8))/32).
    Global interactions are incorporated via a MultiRZ gate with rotation angle given by π multiplied by the sum
    over qubits of (f(i,1)+f(i,2)+f(i,3)+f(i,5)+f(i,6)+2f(i,7)+2f(i,8)) divided by 97.
    Finally, each qubit applies a post‐entanglement block U_post^(i): RZ(π·f(i,7)) → RY(π·f(i,8)) → RX(π·f(i,7)).
    Hardware-aware error mitigation is assumed to be applied at the device level.
    """
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4, global_const: float = 97) -> None:
        """Initialize the Adaptive Fully Connected Global Feature Map v26.
        
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
        """Construct the quantum circuit for the Adaptive Fully Connected Global Feature Map v26.
        
        Args:
            x (np.ndarray): Input normalized feature vector of shape (80,).
        """
        # Step 1: U_local block - encode features f1 to f4 using variable cyclic permutations
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Choose cyclic rotation order based on qubit index modulo 5
            if qubit % 5 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX]
            elif qubit % 5 == 1:
                rotations = [qml.RY, qml.RZ, qml.RX, qml.RY]
            elif qubit % 5 == 2:
                rotations = [qml.RZ, qml.RX, qml.RY, qml.RZ]
            elif qubit % 5 == 3:
                rotations = [qml.RX, qml.RZ, qml.RY, qml.RX]
            else:
                rotations = [qml.RY, qml.RX, qml.RZ, qml.RY]
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(rotations, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor CP gate ring
        for qubit in range(self.n_qubits):
            partner = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, partner])
        
        # Step 3: CRY layer at offset 2 using feature f4
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: CRZ layer at offset 3 using features f5 and f6
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * ((2 * f5 + 3 * f6_partner) / 5.0)
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: CRX layer at offset 4 using feature f2
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Step 6: Extra CRX layer at offset 6 using feature f3
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f3_current = x[base_idx + 2]
            partner = (qubit + 6) % self.n_qubits
            f3_partner = x[8 * partner + 2]
            extra_crx_angle = np.pi * ((f3_current + f3_partner) / 6.0)
            qml.CRX(phi=extra_crx_angle, wires=[qubit, partner])
        
        # Step 7: Ultra-light CRY layer at offset 5 using feature f8 with scaling factor 1/32
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f8_current = x[base_idx + 7]
            partner = (qubit + 5) % self.n_qubits
            f8_partner = x[8 * partner + 7]
            ultra_cry_angle = np.pi * ((f8_current + f8_partner) / 32.0)
            qml.CRY(phi=ultra_cry_angle, wires=[qubit, partner])
        
        # Step 8: Global MultiRZ gate
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
        
        # Step 9: U_post block - fixed cyclic rotations using features f7 and f8
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RY(phi=np.pi * f8, wires=[qubit])
            qml.RX(phi=np.pi * f7, wires=[qubit])
