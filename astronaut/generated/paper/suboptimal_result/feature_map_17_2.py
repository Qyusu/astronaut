import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class StreamlinedUnifiedHybridFeatureMapV17(BaseFeatureMap):
    """
    Streamlined Unified Hybrid Feature Map v17.
    
    This feature map assigns 80 normalized features evenly to 10 qubits (with f_{i,j} = x[8*i + j]).
    Each qubit applies a unified rotation block U_unified that concurrently encodes features f₁, f₂, f₃, f₄, f₇, and f₈ using a fixed cyclic
    permutation, thereby reducing circuit depth by eliminating a separate post-entanglement block.
    Nearest-neighbor entanglement is established via a CP gate ring, followed by a CRY layer at an offset of 2 (using π·((f₄_current + f₄_partner)/2))
    and a CRZ layer at an offset of 3 (using π·(2f₅+3f₆_partner)/5). Global correlations are then consolidated via a MultiRZ
    gate with rotation angle π multiplied by the sum of selected features (f₁+f₂+f₃+f₅+f₆+2f₇+2f₈) divided by 100.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Streamlined Unified Hybrid Feature Map v17.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Streamlined Unified Hybrid Feature Map v17.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,).
        """
        # Step 1: U_unified - Unified rotation block encoding features f₁, f₂, f₃, f₄, f₇, and f₈
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            
            # Choose a fixed cyclic permutation based on the qubit index
            if qubit % 3 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX, qml.RZ, qml.RY]
            elif qubit % 3 == 1:
                rotations = [qml.RY, qml.RZ, qml.RX, qml.RY, qml.RX, qml.RZ]
            else:
                rotations = [qml.RZ, qml.RX, qml.RY, qml.RZ, qml.RY, qml.RX]
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4, np.pi * f7, np.pi * f8]
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
        
        # Step 5: Global entanglement using a MultiRZ gate
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
