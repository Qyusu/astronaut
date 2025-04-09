import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class HybridMultiEntanglementFeatureMapV2(BaseFeatureMap):
    """Hybrid Multi-Entanglement Feature Map v2.
    
    This feature map splits the 80-dimensional input into 10 blocks of 8 features each. For each qubit:
      - A primary local rotation block (U_primary) is applied using the first four features in a cyclic order: RX, RY, RZ, RX.
      - A ring of ControlledPhaseShift (CP) gates is then applied between adjacent qubits.
      - An additional layer of entanglement is introduced via Controlled-RY (CRY) gates connecting qubits separated by three positions; the CRY rotation angle is computed as π times the average of the fifth feature from the connected qubits.
      - Finally, a secondary local rotation block (U_secondary) is applied using the remaining three features in the order: RZ, RX, RY.
    The overall circuit implements |Φ(x)⟩ = (∏ U_secondary) (∏ CRY) (∏ CP) (∏ U_primary)|0⟩.
    """

    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Hybrid Multi-Entanglement Feature Map v2.

        Args:
            n_qubits (int): Number of qubits (should be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        self.cp_angle: float = cp_angle

    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Hybrid Multi-Entanglement Feature Map v2.
        
        The 80 features are divided into 10 blocks of 8 features each. For each qubit i (0 ≤ i ≤ 9):
          1. U_primary: Apply RX, RY, RZ, RX with angles π*f_{i,1}, π*f_{i,2}, π*f_{i,3}, π*f_{i,4} using features 1-4.
          2. Apply a ControlledPhaseShift gate between qubit i and qubit (i+1)%n_qubits with a fixed angle cp_angle.
          3. Apply a CRY gate between qubit i and qubit (i+3)%n_qubits with rotation angle π*((f_{i,5} + f_{(i+3)%n_qubits,5})/2), where f_{i,5} is the fifth feature.
          4. U_secondary: Apply a secondary rotation block in the order RZ, RX, RY with angles π*f_{i,6}, π*f_{i,7}, π*f_{i,8} using features 6-8.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,), with features normalized to [0, 1].
        """
        # Step 1: U_primary local rotations
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            qml.RX(phi=np.pi * f1, wires=[qubit])
            qml.RY(phi=np.pi * f2, wires=[qubit])
            qml.RZ(phi=np.pi * f3, wires=[qubit])
            qml.RX(phi=np.pi * f4, wires=[qubit])
        
        # Step 2: Entanglement via ControlledPhaseShift between adjacent qubits (ring topology)
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Additional entanglement via CRY between qubits separated by three positions
        for qubit in range(self.n_qubits):
            next3 = (qubit + 3) % self.n_qubits
            f5_current = x[8 * qubit + 4]  # fifth feature for current qubit
            f5_next3 = x[8 * next3 + 4]     # fifth feature for qubit at (i+3)%n_qubits
            angle_cry = np.pi * ((f5_current + f5_next3) / 2.0)
            qml.CRY(phi=angle_cry, wires=[qubit, next3])
        
        # Step 4: U_secondary local rotations
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f6 = x[base_idx + 5]
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f6, wires=[qubit])
            qml.RX(phi=np.pi * f7, wires=[qubit])
            qml.RY(phi=np.pi * f8, wires=[qubit])
