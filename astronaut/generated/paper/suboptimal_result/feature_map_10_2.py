import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class UnifiedMultiPermutationGlobalEnrichedFeatureMapV9(BaseFeatureMap):
    """Unified Multi-Permutation Global Enriched Feature Map v9.
    
    This feature map evenly distributes an 80-dimensional (e.g., PCA-reduced) input across 10 qubits, with each qubit i receiving
    features f_{i,j} = x[8*i + j] for j = 0,...,7. A unified rotation block U_unified^(i) encodes features f₁, f₂, f₃, f₄, f₇,
    and f₈ in a single sequence using a unique fixed cyclic permutation determined by the qubit index. This is followed by a
    multilayered entanglement structure: a nearest-neighbor CP gate ring, a CRY layer (offset 2) using features f₄, a CRZ layer (offset 3)
    using features f₅ and f₆, a CRX layer (offset 4) using feature f₂, and an additional CRY layer (offset 5) using feature f₃.
    Global correlations are then integrated by a MultiRZ gate with rotation angle computed as π*(Σᵢ(2f₁ + f₅ + f₆ + 2f₇ + 2f₈))/90.
    
    The overall operation is given by:
      |Φ(x)⟩ = MultiRZ(π*(Σᵢ(2f₁+f₅+f₆+2f₇+2f₈))/90) · (∏ CRY_{i,(i+5) mod 10}(π*((f₃+f₃_partner)/2))) ·
               (∏ CRX_{i,(i+4) mod 10}(π*((f₂+f₂_partner)/2))) · (∏ CRZ_{i,(i+3) mod 10}(π*(2f₅+3f₆_partner)/5)) ·
               (∏ CRY_{i,(i+2) mod 10}(π*((f₄+f₄_partner)/2))) · (∏ CP(q_i, q_{(i+1) mod 10})) · (∏ U_unified^(i)) |0⟩,
    where
      U_unified^(i) = R_{β_{i,1}}(π·f₁) R_{β_{i,2}}(π·f₂) R_{β_{i,3}}(π·f₃) R_{β_{i,4}}(π·f₄)
                         R_{β_{i,5}}(π·f₇) R_{β_{i,6}}(π·f₈), with f_{i,j} = x[8*i + j].
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Unified Multi-Permutation Global Enriched Feature Map v9.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Unified Multi-Permutation Global Enriched Feature Map v9.
        
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
            
            # Select a cyclic permutation from an expanded set based on qubit index
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
        
        # Step 3: Additional entanglement using CRY gates (offset of 2)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: Intermediate entanglement using CRZ gates (offset of 3)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: Further entanglement using CRX gates (offset of 4)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Step 6: Additional entanglement using CRY gates (offset of 5)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f3_current = x[base_idx + 2]
            partner = (qubit + 5) % self.n_qubits
            f3_partner = x[8 * partner + 2]
            cry_angle = np.pi * ((f3_current + f3_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 7: Global entanglement using a MultiRZ gate
        global_sum = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f5 = x[base_idx + 4]
            f6 = x[base_idx + 5]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            global_sum += (2 * f1 + f5 + f6 + 2 * f7 + 2 * f8)
        global_angle = np.pi * (global_sum / 90.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
