import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class UnifiedRotationConsolidatedFeatureMapV8(BaseFeatureMap):
    """Unified Rotation Consolidated Feature Map v8.
    
    This feature map allocates an 80-dimensional input evenly to 10 qubits, with each qubit i receiving
    features f_{i,j} = x[8*i + j] for j = 0,...,7. A unified rotation block, U_unified^(i), encodes features
    f1, f2, f3, f4, f7, and f8 in a single sequence using a cyclic permutation that varies across qubits.
    Nearest-neighbor correlations are established via a ring of ControlledPhaseShift (CP) gates. An additional
    entanglement layer uses CRY gates between qubit i and qubit (i+2) mod 10 with rotation angle
    π*((f2_i + f2_{(i+2)})/2). An intermediate CRZ layer connects qubit i with qubit (i+3) mod 10 with an angle
    π*(2·f5 + 3·f6_partner)/5. Finally, a global MultiRZ gate is applied with a rotation angle computed as
    π*(Σ_{i}(f5 + 2·f7 + 2·f8))/50. This design consolidates the rotation stages to reduce circuit depth.
    
    The overall circuit implements:
      |Φ(x)⟩ = (∏_{i=0}^{9} U_unified^(i)) · (∏_{i=0}^{9} CP(q_i, q_{(i+1) mod 10})) ·
               (∏_{i=0}^{9} CRY_{i,(i+2) mod 10}(π*((f2 + f2_partner)/2))) ·
               (∏_{i=0}^{9} CRZ_{i,(i+3) mod 10}(π*(2f5+3f6)/5)) ·
               MultiRZ(π*(Σ_{i}(f5+2f7+2f8))/50) |0⟩.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Unified Rotation Consolidated Feature Map v8.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Unified Rotation Consolidated Feature Map v8.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,).
        """
        # Step 1: U_unified - Apply a unified rotation block encoding features f1, f2, f3, f4, f7, and f8
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            
            # Choose a cyclic permutation of rotation gates based on qubit index
            if qubit % 3 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX, qml.RZ, qml.RY]
            elif qubit % 3 == 1:
                rotations = [qml.RY, qml.RZ, qml.RX, qml.RY, qml.RX, qml.RZ]
            else:
                rotations = [qml.RZ, qml.RX, qml.RY, qml.RZ, qml.RY, qml.RX]
            
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4, np.pi * f7, np.pi * f8]
            for gate, angle in zip(rotations, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Additional entanglement using CRY gates between qubit i and (i+2) mod n_qubits
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 2) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            angle_cry = np.pi * ((f2_current + f2_partner) / 2.0)
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
