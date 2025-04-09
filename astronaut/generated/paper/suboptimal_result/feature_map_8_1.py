import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class WeightedGlobalAdditionalEntanglementFeatureMapV8(BaseFeatureMap):
    """Weighted Global and Additional Entanglement Feature Map v8.
    
    This feature map evenly assigns the 80-dimensional input to 10 qubits, where each qubit i receives
    features f_{i,j} = x[8*i + j] for j = 0,...,7. The circuit applies the following operations:
      - U_local: A variable local rotation block is applied on each qubit using features f1–f4 with a cyclic
                 permutation that varies with the qubit index. Formally, U_local^(i) = R_{α_{i,1}}(π·f1) ·
                 R_{α_{i,2}}(π·f2) · R_{α_{i,3}}(π·f3) · R_{α_{i,4}}(π·f4).
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates is applied between qubit i and
                 qubit (i+1) mod 10.
      - Additional entanglement: CRY gates are applied between qubit i and qubit (i+2) mod 10, with rotation angle
                 π multiplied by the average of the 4th feature from both qubits.
      - Intermediate entanglement: CRZ gates connect qubit i with qubit (i+3) mod 10 with rotation angles
                 computed as π*(2·f5 + 3·f6_partner)/5, where f5 comes from qubit i and f6_partner from qubit (i+3).
      - Global entanglement: A MultiRZ gate is applied over all qubits with a rotation angle defined as
                 π multiplied by the weighted sum over all qubits of (f5 + 2·f7 + 2·f8) divided by 50.
      - U_post: A post-entanglement rotation block is applied on each qubit that implements RX followed by RZ using
                 features f8 and f7 respectively.
    
    The overall circuit implements:
      |Φ(x)⟩ = (∏_{i=0}^{9} U_post^(i)) · MultiRZ(π·(Σ_{i}(f5 + 2f7 + 2f8))/50) · (∏_{i=0}^{9} CRZ_{i,(i+3) mod 10}(π·(2f5+3f6)/5)) ·
               (∏_{i=0}^{9} CRY_{i,(i+2) mod 10}(π·((f4_i+f4_{(i+2)})/2))) · (∏_{i=0}^{9} CP(q_i, q_{(i+1) mod 10})) · (∏_{i=0}^{9} U_local^(i)) |0⟩.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Weighted Global and Additional Entanglement Feature Map v8.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Weighted Global and Additional Entanglement Feature Map v8.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,).
        """
        # Step 1: U_local - Apply a variable local rotation block using features f1–f4 on each qubit
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Select a cyclic permutation of rotation gates based on qubit index
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
        
        # Step 3: Additional entanglement using CRY gates between qubit i and (i+2) mod n_qubits
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            angle_cry = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=angle_cry, wires=[qubit, partner])
        
        # Step 4: Intermediate entanglement using CRZ gates between qubit i and (i+3) mod n_qubits
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            angle_crz = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 5: Global entanglement using MultiRZ across all qubits
        weighted_sum = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            weighted_sum += (f5 + 2 * f7 + 2 * f8)
        global_angle = np.pi * (weighted_sum / 50.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 6: U_post - Apply post-entanglement rotations using features f7 and f8
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RZ(phi=np.pi * f7, wires=[qubit])
