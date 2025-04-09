import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class OptimizedExpandedPermutationGlobalFeatureMapV14(BaseFeatureMap):
    """Optimized Expanded Permutation Global Feature Map v14.
    
    This feature map evenly allocates an 80-dimensional normalized input to 10 qubits, with each qubit i receiving
    features f_{i,1} through f_{i,8} (with f_{i,j} = x[8*i + j]). A unified rotation block U_unified^(i) encodes features
    f₁, f₂, f₃, f₄, f₇, and f₈ concurrently using fixed rotations determined by a cyclic permutation chosen from an
    expanded repertoire. This reduces circuit depth while maximizing state diversity.
    
    The entanglement strategy is layered:
      - A nearest-neighbor CP gate ring entangles adjacent qubits.
      - A CRY layer (offset 2) uses the average of f₄ for local entanglement.
      - A CRZ layer (offset 3) applies rotations based on 2f₅+3f₆ scaled by 1/5.
      - A CRX layer (offset 4) mixes states via the average of f₂.
      - An extra CRX layer (offset 6) uses f₃ to capture secondary correlations.
      - An ultra-light CRZ layer (offset 5) taps additional correlations using f₈ with a toned-down angle.
      - Global interactions are integrated with a MultiRZ gate whose rotation angle is π multiplied by the sum of
        (f₁+f₂+f₃+f₅+f₆+2f₇+2f₈) over all qubits, normalized by 105.
    
    The unified rotation block U_unified^(i) is defined as:
      U_unified^(i) = R_{β_{i,1}}(π f_{i,1}) R_{β_{i,2}}(π f_{i,2}) R_{β_{i,3}}(π f_{i,3})
                       R_{β_{i,4}}(π f_{i,4}) R_{β_{i,5}}(π f_{i,7}) R_{β_{i,6}}(π f_{i,8}).
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Optimized Expanded Permutation Global Feature Map v14.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Optimized Expanded Permutation Global Feature Map v14.
        
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
            
            # Choose a fixed cyclic permutation from an expanded set based on qubit index modulo 3
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
        
        # Step 3: CRY layer (offset 2) using the average of feature f₄
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: CRZ layer (offset 3) using features f₅ and f₆
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: CRX layer (offset 4) using the average of feature f₂
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Step 6: Extra CRX layer (offset 6) using feature f₃ for secondary correlations
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f3_current = x[base_idx + 2]
            partner = (qubit + 6) % self.n_qubits
            f3_partner = x[8 * partner + 2]
            extra_crx_angle = np.pi * ((f3_current + f3_partner) / 6.0)
            qml.CRX(phi=extra_crx_angle, wires=[qubit, partner])
        
        # Step 7: Ultra-light CRZ layer (offset 5) using feature f₈ with a toned-down rotation angle
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f8_current = x[base_idx + 7]
            partner = (qubit + 5) % self.n_qubits
            f8_partner = x[8 * partner + 7]
            crz_ul_angle = np.pi * ((f8_current + f8_partner) / 10.0)
            qml.CRZ(phi=crz_ul_angle, wires=[qubit, partner])
        
        # Step 8: Global entanglement using a MultiRZ gate
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
