import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class RecalibratedTwoBlockHybridFeatureMapV20(BaseFeatureMap):
    """
    Recalibrated Two-Block Hybrid Feature Map v20 (Adaptive Augmentation).
    
    This feature map partitions an 80-dimensional normalized input among 10 qubits (with f(i,j) = x[8*i + j]).
    Each qubit applies a local rotation block U_local^(i) that encodes features f₁–f₄ using variable cyclic orders chosen from an expanded set (using a mod 2 pattern).
    Nearest-neighbor coupling is established via a sequential ControlledPhaseShift (CP) gate ring.
    The entanglement layers consist of:
      - An offset-2 CRY layer with rotation π*((f₄_current + f₄_partner)/2).
      - An offset-3 CRZ layer with rotation π*(2f₅ + 3f₆_partner)/5.
      - An offset-4 CRX layer with rotation π*((f₂_current + f₂_partner)/2).
      - An extra offset-6 CRX layer with a reduced rotation π*((f₃_current + f₃_partner)/6).
    Global correlations are merged via a MultiRZ gate with rotation angle π*(global sum/95).
    Finally, the post-entanglement block U_post^(i) applies a fixed cyclic sequence: RZ(π·f₇) → RX(π·f₈) → RY(π·f₇) and is augmented by an extra RY(π/4).
    
    This modular design facilitates parallel execution of commuting operations and enhances noise robustness.
    """
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Recalibrated Two-Block Hybrid Feature Map v20 (Adaptive Augmentation).
        
        Args:
            n_qubits (int): Number of qubits (should be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default: π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle

    def feature_map(self, x: np.ndarray) -> None:
        """Construct the quantum circuit for the Recalibrated Two-Block Hybrid Feature Map v20 (Adaptive Augmentation).
        
        Args:
            x (np.ndarray): Input normalized feature vector of shape (80,).
        """
        # Step 1: U_local block - encode features f₁ to f₄ with a variable cyclic order based on qubit index (mod 2 pattern)
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
        
        # Step 2: Nearest-neighbor entanglement via a sequential CP gate ring
        for qubit in range(self.n_qubits):
            partner = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, partner])
        
        # Step 3: Offset-2 CRY layer (using feature f₄)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f4_current = x[base_idx + 3]
            partner = (qubit + 2) % self.n_qubits
            f4_partner = x[8 * partner + 3]
            cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
            qml.CRY(phi=cry_angle, wires=[qubit, partner])
        
        # Step 4: Offset-3 CRZ layer (using features f₅ and f₆)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            f6_partner = x[8 * partner + 5]
            crz_angle = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=crz_angle, wires=[qubit, partner])
        
        # Step 5: Offset-4 CRX layer (using feature f₂)
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
        
        # Step 7: Global entanglement using MultiRZ
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
        
        # Step 8: U_post block - apply fixed cyclic rotations and an extra RY(π/4) augmentation
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
            qml.RY(phi=np.pi / 4, wires=[qubit])


