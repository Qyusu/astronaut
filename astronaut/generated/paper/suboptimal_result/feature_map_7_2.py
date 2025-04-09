import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class EnhancedPostEntangledVariableRotationFeatureMapV7(BaseFeatureMap):
    """Enhanced Post-Entangled Variable Rotation Feature Map v7.
    
    This feature map allocates the complete 80-dimensional PCA-reduced input evenly among 10 qubits,
    where each qubit i (0 ≤ i ≤ 9) receives features f_{i,j} = x[8*i + j] for j = 0,...,7. An additional
    normalization step is applied to ensure that rotation angles remain within the optimal operational range.
    The circuit applies the following operations:
      - U_local: A local rotation block encodes features f₁–f₄ using a cyclic order that varies between qubits.
                 For even-indexed qubits the order is [RX, RY, RZ, RX] and for odd-indexed qubits it is [RY, RZ, RX, RY].
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates entangles adjacent qubits.
      - Intermediate entanglement: CRZ gates connect qubit i with qubit (i+3) mod 10 with rotation angles
                 set as π*(2·f_{i,5}+3·f_{(i+3),6})/5, capturing medium-range feature dependencies.
      - Global entanglement: A MultiRZ gate is applied over all qubits with rotation angle defined as
                 π*((∑_{i=0}^{9}(f_{i,7}+f_{i,8}))/20).
      - U_post: A post-entanglement rotation block applies RX followed by RZ using features f₈ and f₇ respectively
                 to further diversify the quantum state.
    
    The overall circuit implements:
      |Φ(x)⟩ = (∏ U_post) · MultiRZ(π*(Σ(f₇+f₈)/20)) · (∏ CRZ_{i,(i+3) mod 10}) · (∏ CP) · (∏ U_local) |0⟩.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Enhanced Post-Entangled Variable Rotation Feature Map v7.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Enhanced Post-Entangled Variable Rotation Feature Map v7.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,). This vector will be normalized to [0, 1].
        """
        # Normalize features to [0, 1] to ensure rotation angles remain within optimal range
        x_min = np.min(x)
        x_max = np.max(x)
        x_norm = (x - x_min) / (x_max - x_min + 1e-10)
        
        # Step 1: U_local - Local rotation block using normalized features f₁–f₄
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x_norm[base_idx + 0]
            f2 = x_norm[base_idx + 1]
            f3 = x_norm[base_idx + 2]
            f4 = x_norm[base_idx + 3]
            
            # Use a cyclic order based on qubit index (even vs odd)
            if qubit % 2 == 0:
                local_order = [qml.RX, qml.RY, qml.RZ, qml.RX]
            else:
                local_order = [qml.RY, qml.RZ, qml.RX, qml.RY]
            
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
            f5 = x_norm[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x_norm[8 * partner + 5]
            angle_crz = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 4: Global entanglement using MultiRZ across all qubits
        total = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x_norm[base_idx + 6]
            f8 = x_norm[base_idx + 7]
            total += (f7 + f8)
        global_angle = np.pi * (total / 20.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 5: U_post - Post-entanglement rotation block applying RX then RZ using features f₈ and f₇
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x_norm[base_idx + 6]
            f8 = x_norm[base_idx + 7]
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RZ(phi=np.pi * f7, wires=[qubit])
