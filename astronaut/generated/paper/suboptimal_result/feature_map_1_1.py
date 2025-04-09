import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class LayeredRotationalEntanglementFeatureMap(BaseFeatureMap):
    """Layered Rotational-Entanglement Feature Map.

    This feature map partitions the 80-dimensional input into 10 blocks of 8 features each.
    Each qubit undergoes a sequence of single-qubit rotations (RX, RY, RZ in a cyclic order),
    with rotation angles computed as π times the corresponding input feature (assumed normalized to [0,1]).
    After performing these rotations, a ring of ControlledPhaseShift gates is applied between neighboring qubits
    to introduce entanglement and capture global correlations.
    """

    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4) -> None:
        """Initialize the Layered Rotational-Entanglement Feature Map.

        Args:
            n_qubits (int): Number of qubits (should be 10 for 80-dimensional input).
            cp_angle (float): Phase angle for the ControlledPhaseShift entangling gates (default π/4).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        self.cp_angle: float = cp_angle

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of the layered rotational-entanglement feature map.

        The 80 features are divided into 10 blocks of 8 features each; each block is encoded on a qubit using a
        cyclic pattern of single-qubit rotations (RX, RY, RZ, ...). The rotation angle for each gate is computed as
        π * (feature value). After encoding, ControlledPhaseShift gates are applied between each qubit and its
        neighbor (in a ring) with a fixed phase shift to induce entanglement.

        Args:
            x (np.ndarray): Input data of shape (80,), assumed to be normalized to [0, 1].
        """
        # Encode each block of 8 features into a qubit using cyclic rotations
        for qubit in range(self.n_qubits):
            # Extract the block of 8 features for the current qubit
            block = x[8 * qubit : 8 * (qubit + 1)]
            for j, feature in enumerate(block):
                angle = np.pi * feature  # Scale the feature by π
                # Apply rotations in cyclic order: RX, RY, RZ, ...
                if j % 3 == 0:
                    qml.RX(phi=angle, wires=[qubit])
                elif j % 3 == 1:
                    qml.RY(phi=angle, wires=[qubit])
                elif j % 3 == 2:
                    qml.RZ(phi=angle, wires=[qubit])

        # Apply a ring of ControlledPhaseShift gates to entangle neighboring qubits
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
