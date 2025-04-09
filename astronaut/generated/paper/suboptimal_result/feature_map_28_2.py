import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class HierarchicalBridgingEnhancedAdaptiveFeatureMapV27(BaseFeatureMap):
    """
    Hierarchical Bridging Enhanced Adaptive Feature Map v27.
    
    This feature map partitions an 80-dimensional normalized input among 10 qubits, which are divided into two sub-blocks
    (Block A: qubits 0–4 and Block B: qubits 5–9). Each qubit applies a local rotation block U_local^(i) that encodes features f₁–f₄
    using fixed cyclic orders. Within each sub-block, intra-block entanglement is applied via:
      - A CP gate ring connecting nearest neighbors within the block.
      - A CRY layer at offset 2 applying rotations with angle π*((f₍i,4₎ + f₍(i+2),4₎)/2).
      - A CRZ layer at offset 3 applying rotations with angle π*((2f₍i,5₎ + 3f₍(i+3),6₎)/5).
      - A CRX layer at offset 4 applying rotations with angle π*((f₍i,2₎ + f₍(i+4),2₎)/2).
      - An additional CRX layer at offset 6 (using modulo arithmetic) applying rotations with angle π*((f₍i,3₎ + f₍(i+6),3₎)/6).
    Inter-block bridging is achieved via CRZ gates connecting corresponding qubits from Block A and Block B
    with rotation angle π*((f₍A,5₎ + f₍B,5₎)/4). Global correlations are integrated via a MultiRZ gate with angle
    π * (Σ(f₁+f₂+f₃+f₅+f₆+2f₇+2f₈))/97. Finally, the U_post block applies a fixed cyclic sequence of rotations:
    RZ(π·f₇) → RX(π·f₈) → RY(π·f₇) followed by an additional RY(π/4).
    
    All gate parameters are linear functions of the input features.
    """
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4, global_const: float = 97) -> None:
        """Initialize the Hierarchical Bridging Enhanced Adaptive Feature Map v27.
        
        Args:
            n_qubits (int): Number of qubits (should be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for ControlledPhaseShift gates (default: π/4).
            global_const (float): Normalization constant for the MultiRZ gate (default: 97).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        self.global_const = global_const
        self.block_size = n_qubits // 2  # Divide 10 qubits into two blocks of 5
        
    def feature_map(self, x: np.ndarray) -> None:
        """Construct the quantum circuit for the Hierarchical Bridging Enhanced Adaptive Feature Map v27.
        
        Args:
            x (np.ndarray): Input normalized feature vector of shape (80,).
        """
        # Step 1: U_local block - encode features f1 to f4 for each qubit
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Fixed cyclic rotation order based on qubit index modulo 5
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
        
        # Define sub-blocks: Block A (qubits 0–4) and Block B (qubits 5–9)
        block_A = list(range(0, self.block_size))
        block_B = list(range(self.block_size, self.n_qubits))
        
        # Step 2: Intra-block entanglement for each block
        for block in [block_A, block_B]:
            # (a) Apply CP gate ring within the block (nearest neighbors)
            for i in range(len(block)):
                partner = block[(i + 1) % self.block_size]
                qml.ControlledPhaseShift(phi=self.cp_angle, wires=[block[i], partner])
            
            # (b) CRY layer at offset 2 within the block
            for i in range(len(block)):
                current = block[i]
                partner = block[(i + 2) % self.block_size]
                base_current = 8 * current
                base_partner = 8 * partner
                f4_current = x[base_current + 3]
                f4_partner = x[base_partner + 3]
                cry_angle = np.pi * ((f4_current + f4_partner) / 2.0)
                qml.CRY(phi=cry_angle, wires=[current, partner])
            
            # (c) CRZ layer at offset 3 within the block
            for i in range(len(block)):
                current = block[i]
                partner = block[(i + 3) % self.block_size]
                base_current = 8 * current
                base_partner = 8 * partner
                f5_current = x[base_current + 4]
                f6_partner = x[base_partner + 5]
                crz_angle = np.pi * ((2 * f5_current + 3 * f6_partner) / 5.0)
                qml.CRZ(phi=crz_angle, wires=[current, partner])
            
            # (d) CRX layer at offset 4 within the block
            for i in range(len(block)):
                current = block[i]
                partner = block[(i + 4) % self.block_size]
                base_current = 8 * current
                base_partner = 8 * partner
                f2_current = x[base_current + 1]
                f2_partner = x[base_partner + 1]
                crx_angle = np.pi * ((f2_current + f2_partner) / 2.0)
                qml.CRX(phi=crx_angle, wires=[current, partner])
            
            # (e) Additional CRX layer at offset 6 (using modulo arithmetic) within the block
            # In a block of size 5, (i+6) mod 5 is equivalent to (i+1) mod 5
            for i in range(len(block)):
                current = block[i]
                partner = block[(i + 6) % self.block_size]
                base_current = 8 * current
                base_partner = 8 * partner
                f3_current = x[base_current + 2]
                f3_partner = x[base_partner + 2]
                extra_crx_angle = np.pi * ((f3_current + f3_partner) / 6.0)
                qml.CRX(phi=extra_crx_angle, wires=[current, partner])
        
        # Step 3: Inter-block bridging using CRZ gates between corresponding qubits
        for i in range(self.block_size):
            qubit_A = block_A[i]
            qubit_B = block_B[i]
            base_A = 8 * qubit_A
            base_B = 8 * qubit_B
            f5_A = x[base_A + 4]
            f5_B = x[base_B + 4]
            bridge_angle = np.pi * ((f5_A + f5_B) / 4.0)
            qml.CRZ(phi=bridge_angle, wires=[qubit_A, qubit_B])
        
        # Step 4: Global MultiRZ gate
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
        
        # Step 5: U_post block - apply fixed cyclic rotations: RZ, RX, RY and an extra RY(π/4)
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
            qml.RY(phi=np.pi / 4, wires=[qubit])
