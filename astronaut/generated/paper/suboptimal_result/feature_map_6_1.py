import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class SimplifiedGlobalIntermediateEntanglementFeatureMapV6(BaseFeatureMap):
    """Simplified Global and Intermediate Entanglement Feature Map v6.
    
    This feature map partitions the 80-dimensional input into 10 blocks of 8 features each.
    For each qubit i (0 ≤ i ≤ 9), with features f_{i,j} = x[8*i + j] (for j = 0,...,7):
      - U_local: A single local rotation block is applied with a cyclic permutation of gates that varies
                 across qubits. For example, if i mod 3 == 0, the gate order is [RX, RY, RZ, RX];
                 if i mod 3 == 1, the order is [RY, RZ, RX, RY]; and if i mod 3 == 2, the order is [RZ, RX, RY, RZ].
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates is applied between qubit i
                 and qubit (i+1) mod 10.
      - Intermediate entanglement: CRZ gates are applied between qubit i and qubit (i+3) mod 10 with rotation
                 angles computed as π*(2*f_{i,5} + 3*f_{(i+3),6})/5, where f_{i,5} is the 5th feature of qubit i
                 and f_{(i+3),6} is the 6th feature of the partner qubit.
      - Global entanglement: A MultiRZ gate is applied over all qubits with rotation angle defined as
                 π*((Σ_{i=0}^{9} (f_{i,7}+f_{i,8}))/20), thereby integrating global feature interactions.
    
    The overall circuit implements:
      |Φ(x)⟩ = (∏_{i=0}^{9} U^{(i)}_{local}) · (∏_{i=0}^{9} CP(q_i, q_{(i+1) mod 10}) ) ·
              (∏_{i=0}^{9} CRZ_{i,(i+3) mod 10}(π*(2f_{i,5}+3f_{(i+3),6})/5)) ·
              MultiRZ(π*(Σ_{i=0}^{9}(f_{i,7}+f_{i,8}))/20) |0⟩.
    
    All operations are fixed and linear ensuring full data fidelity for robust QSVM classification.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Simplified Global and Intermediate Entanglement Feature Map v6.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Simplified Global and Intermediate Entanglement Feature Map v6.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,), with features normalized to [0, 1].
        """
        # Step 1: Local rotation block (U_local) with a cyclic permutation that varies with qubit index
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Vary the cyclic order based on qubit index
            if qubit % 3 == 0:
                local_order = [qml.RX, qml.RY, qml.RZ, qml.RX]
            elif qubit % 3 == 1:
                local_order = [qml.RY, qml.RZ, qml.RX, qml.RY]
            else:
                local_order = [qml.RZ, qml.RX, qml.RY, qml.RZ]
            
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(local_order, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Intermediate entanglement using CRZ gates between qubit i and (i+3) mod n_qubits
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            # f6 is taken from the partner qubit's 6th feature (index 5)
            f6_partner = x[8 * partner + 5]
            angle_crz = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 4: Global entanglement using MultiRZ across all qubits
        total = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            total += (f7 + f8)
        global_angle = np.pi * (total / 20.0)  
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))

