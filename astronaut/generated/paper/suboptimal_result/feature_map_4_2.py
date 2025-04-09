import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class DiverseLocalGlobalHybridFeatureMapV4(BaseFeatureMap):
    """Diverse Local & Global Hybrid Feature Map v4.
    
    This feature map partitions the 80-dimensional input into 10 blocks of 8 features each.
    For each qubit i (0 ≤ i ≤ 9), with features f_{i,j} = x[8*i + j]:
      - U_primary: A primary local rotation block is applied with a varied cyclic order depending on qubit index.
        For instance, if i is even, the order is [RX, RY, RZ, RX]; if odd, [RY, RZ, RX, RY].
        These rotations use features f1–f4 (indices 0–3) scaled by π.
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates entangles qubit i and (i+1) mod 10.
      - Intermediate entanglement: CRZ gates are applied between qubit i and qubit (i+3) mod 10 with angle
        π * (2*f_{i,5} + 3*f_{(i+3) mod 10,6})/5, where f_{i,5} is from index 4 and f_{(i+3) mod 10,6} from index 5.
      - Tertiary entanglement: CRY gates are applied between qubit i and qubit (i+4) mod 10 with angle
        π * (f_{i,6} - 2*f_{(i+4) mod 10,5})/3, with f_{i,6} from index 5 and f_{(i+4) mod 10,5} from index 4.
      - Global entanglement: A MultiRZ gate is applied over all qubits with the rotation angle set to π times the
        average of (f_{i,7} + f_{i,8})/2 across all qubits.
      - U_post: A post-entanglement rotation block is applied in the order RX, RZ, RY using features f_{i,8} and f_{i,7}.
        Specifically, U_post^(i) = RX(π f_{i,8}) · RZ(π f_{i,7}) · RY(π f_{i,8}).
    An optional post-processing step inspired by Bit Flip Tolerance (BFT) can be applied externally to the measurement
    outcomes to enhance noise robustness.
    
    The overall circuit implements
      |Φ(x)⟩ = (∏ U_post^(i)) · MultiRZ(π ((∑_i (f_{i,7}+f_{i,8}))/20)) · (∏ CRY_{i,(i+4) mod 10}(π(f_{i,6} - 2 f_{(i+4) mod 10,5})/3)) ·
              (∏ CRZ_{i,(i+3) mod 10}(π(2 f_{i,5} + 3 f_{(i+3) mod 10,6})/5)) ·
              (∏ CP(q_i, q_{(i+1) mod 10})) · (∏ U_primary^(i)) |0⟩.
    """

    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Diverse Local & Global Hybrid Feature Map v4.
        
        Args:
            n_qubits (int): Number of qubits (should be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle

    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Diverse Local & Global Hybrid Feature Map v4.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,), with features normalized to [0, 1].
        """
        # Step 1: U_primary local rotations with varied cyclic order
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Vary the rotation sequence based on whether the qubit index is even or odd
            if qubit % 2 == 0:
                primary_order = [qml.RX, qml.RY, qml.RZ, qml.RX]
            else:
                primary_order = [qml.RY, qml.RZ, qml.RX, qml.RY]
            
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(primary_order, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Intermediate entanglement using CRZ gates between qubit i and (i+3) mod n_qubits
        for qubit in range(self.n_qubits):
            partner = (qubit + 3) % self.n_qubits
            f5 = x[8 * qubit + 4]
            f6_partner = x[8 * partner + 5]
            angle_crz = np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 4: Tertiary entanglement using CRY gates between qubit i and (i+4) mod n_qubits
        for qubit in range(self.n_qubits):
            partner = (qubit + 4) % self.n_qubits
            f6 = x[8 * qubit + 5]
            f5_partner = x[8 * partner + 4]
            angle_cry = np.pi * (f6 - 2 * f5_partner) / 3.0
            qml.CRY(phi=angle_cry, wires=[qubit, partner])
        
        # Step 5: Global entanglement using MultiRZ across all qubits
        total = 0.0
        for qubit in range(self.n_qubits):
            f7 = x[8 * qubit + 6]
            f8 = x[8 * qubit + 7]
            total += (f7 + f8)
        global_angle = np.pi * (total / (20.0))  # since sum over (f7+f8)/20
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
        
        # Step 6: U_post local rotations in the order RX, RZ, RY using features f8 and f7
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RY(phi=np.pi * f8, wires=[qubit])
        
        # Note: An optional post-processing step inspired by Bit Flip Tolerance (BFT) may be applied
        # externally on the measurement outcomes to enhance noise robustness.
