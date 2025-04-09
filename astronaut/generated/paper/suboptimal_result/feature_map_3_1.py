import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class TripleEntanglementFeatureMapV3(BaseFeatureMap):
    """Triple-Entanglement Feature Map v3.
    
    This feature map partitions the 80-dimensional input into 10 blocks of 8 features each.
    For each qubit:
      - U_primary: Apply RX, RY, RZ, RX with angles π*f₁, π*f₂, π*f₃, π*f₄ using features 1-4.
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates entangles adjacent qubits.
      - Secondary entanglement: Apply CRZ gates between each qubit and its next-nearest neighbor with rotation angle
        π * ((2 * f₅ + f₅_partner) / 3), where f₅ is the fifth feature of the current qubit and f₅_partner is that of the
        qubit at (i+2) mod 10.
      - Tertiary entanglement: Apply CRX gates between each qubit and the qubit four positions ahead with rotation angle
        π * ((2 * f₆ - f₆_partner) / 3), where f₆ is the sixth feature of the current qubit and f₆_partner is that of
        the qubit at (i+4) mod 10.
      - U_post: A post-entanglement rotation block applies local rotations in the order RZ, RX, RY using features f₇, f₈, and
        reusing f₇.
    The overall circuit implements |Φ(x)⟩ = (∏ U_post) (∏ CRX) (∏ CRZ) (∏ CP) (∏ U_primary)|0⟩.
    """

    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Triple-Entanglement Feature Map v3.
        
        Args:
            n_qubits (int): Number of qubits (should be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle

    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Triple-Entanglement Feature Map v3.
        
        The 80 features are divided into 10 blocks of 8 features each. For each qubit i (0 ≤ i ≤ 9):
          1. U_primary: Apply RX, RY, RZ, RX with angles π*f₁, π*f₂, π*f₃, π*f₄ (features 1-4).
          2. Apply a ControlledPhaseShift gate between qubit i and qubit (i+1)%n_qubits with fixed angle cp_angle.
          3. Secondary entanglement: Apply a CRZ gate between qubit i and qubit (i+2)%n_qubits with angle:
             π * ((2 * f₅ + f₅_partner) / 3), where f₅ is the fifth feature of qubit i and f₅_partner is the fifth feature
             of qubit (i+2)%n_qubits.
          4. Tertiary entanglement: Apply a CRX gate between qubit i and qubit (i+4)%n_qubits with angle:
             π * ((2 * f₆ - f₆_partner) / 3), where f₆ is the sixth feature of qubit i and f₆_partner is that of qubit (i+4)%n_qubits.
          5. U_post: Apply local rotations RZ, RX, RY with angles π*f₇, π*f₈, π*f₇ (features 7, 8, and 7 reused).
        
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
        
        # Step 2: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Secondary entanglement using CRZ gates between next-nearest neighbors
        for qubit in range(self.n_qubits):
            partner = (qubit + 2) % self.n_qubits
            f5_current = x[8 * qubit + 4]
            f5_partner = x[8 * partner + 4]
            angle_crz = np.pi * ((2 * f5_current + f5_partner) / 3.0)
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 4: Tertiary entanglement using CRX gates between qubits separated by four positions
        for qubit in range(self.n_qubits):
            partner = (qubit + 4) % self.n_qubits
            f6_current = x[8 * qubit + 5]
            f6_partner = x[8 * partner + 5]
            angle_crx = np.pi * ((2 * f6_current - f6_partner) / 3.0)
            qml.CRX(phi=angle_crx, wires=[qubit, partner])
        
        # Step 5: U_post local rotations
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            qml.RZ(phi=np.pi * f7, wires=[qubit])
            qml.RX(phi=np.pi * f8, wires=[qubit])
            qml.RY(phi=np.pi * f7, wires=[qubit])
