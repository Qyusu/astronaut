import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class RecalibratedTwoBlockHybridFeatureMapV22(BaseFeatureMap):
    """
    Recalibrated Two-Block Hybrid Feature Map with Ultra-Light CRY at Offset 7 v22.
    
    This feature map partitions an 80-dimensional normalized input among 10 qubits (with f_{i,j} = x[8*i + j]).
    Each qubit applies a dedicated local rotation block U_local^(i) that encodes features f₁–f₄ using variable cyclic
    permutations drawn from an expanded set. Nearest-neighbor coupling is established via a CP gate ring, followed by:
      - A CRY layer at offset 2 applying π·((f₄_current + f₄_partner)/2).
      - A CRZ layer at offset 3 applying π·(2f₅ + 3f₆_partner)/5.
      - A CRX layer at offset 4 applying π·((f₂_current + f₂_partner)/2).
      - An extra CRX layer at offset 6 applying a reduced rotation π·((f₃_current + f₃_partner)/6).
      - An ultra-light CRY layer at offset 7 applying π·((f₈_current + f₈_partner)/40), with partner determined by (i+7) mod 10.
    Global correlations are merged via a MultiRZ gate with rotation angle π·(Σ(f₁+f₂+f₃+f₅+f₆+2f₇+2f₈)/95).
    Finally, the post-entanglement block U_post^(i) applies a fixed cyclic sequence: RZ(π·f₇) → RX(π·f₈) → RY(π·f₇) followed
    by an additional fixed RY(π/4) rotation to further diversify the state.
    
    This modular design facilitates the integration of hardware-aware error mitigation techniques such as dynamical
    decoupling and Bit Flip Tolerance for improved resilience on NISQ devices.
    """
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Recalibrated Two-Block Hybrid Feature Map with Ultra-Light CRY at Offset 7 v22.
        
        Args:
            n_qubits (int): Number of qubits (should be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default: π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        
    def feature_map(self, x: np.ndarray) -> None:
        """Construct the quantum circuit for the Recalibrated Two-Block Hybrid Feature Map with Ultra-Light CRY at Offset 7 v22.
        
        Args:
            x (np.ndarray): Input normalized feature vector of shape (80,).
        """
        # Step 1: U_local block - encode features f₁ to f₄ with a variable cyclic order (using mod 2 for diversity)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            if qubit % 2 == 0:
                rotations = [qml.RX, qml.RY, qml.RZ, qml.RX]
            else:
                rotations = [qml.RY, qml.RZ, qml.RX, qml.RY]
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(rotations, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement via a CP gate ring
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: CRY layer at offset 2 (using feature f₄)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: CRZ layer at offset 3 (using features f₅ and f₆)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: CRX layer at offset 4 (using feature f₂)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f2_current = x[base_idx + 1]
            partner = (qubit + 4) % self.n_qubits
            f2_partner = x[8 * partner + 1]
            crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
            qml.CRX(phi=crx_angle, wires=[qubit, partner])
        
        # Step 6: Extra CRX layer at offset 6 (using feature f₃)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f3_current = x[base_idx + 2]
            partner = (qubit + 6) % self.n_qubits
            f3_partner = x[8 * partner + 2]
            extra_crx_angle = np.pi * ((f3_current + f3_partner) / 6.0)
            qml.CRX(phi=extra_crx_angle, wires=[qubit, partner])
        
        # Step 7: Ultra-light CRY layer at offset 7 (using feature f₈) with a scaling denominator of 40
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f8_current = x[base_idx + 7]
            partner = (qubit + 7) % self.n_qubits
            f8_partner = x[8 * partner + 7]
            ultra_cry_angle = np.pi * ((f8_current + f8_partner) / 40.0)
            qml.CRY(phi=ultra_cry_angle, wires=[qubit, partner])
        
        # Step 8: Global entanglement using MultiRZ
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
        global_angle = np.pi * (global_sum / 95.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 9: U_post block - apply a fixed cyclic rotation sequence enhanced by an extra RY(π/4)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
            qml.RY(phi=np.pi / 4, wires=[qubit])
