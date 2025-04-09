import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class ModularBlockCRotIsingFeatureMap(BaseFeatureMap):
    """Modular Block CRot-Ising Feature Map.

    This feature map splits the 80-dimensional input into 10 blocks of 8 features each.
    Initially, Hadamard gates are applied to all qubits to generate superposition.
    Each qubit then receives an RX rotation, where the rotation angle is computed as (π/8) times the sum
    of the features in its corresponding block. Next, CRot gates are applied between adjacent qubits
    based on an overlapping subset of features (last 2 features of the current block and first 2 features
    of the next block), with the computed overlapping angle used for all parameters of the CRot gate.
    Finally, an entanglement layer is added using IsingXX and IsingZZ gates in a nearest-neighbor ring configuration.
    """

    def __init__(self, n_qubits: int, ising_angle_xx: float = np.pi/8, ising_angle_zz: float = np.pi/8) -> None:
        """Initialize the Modular Block CRot-Ising Feature Map.

        Args:
            n_qubits (int): Number of qubits (should be 10 for 80-dimensional input).
            ising_angle_xx (float): Rotation angle for the IsingXX gates (default π/8).
            ising_angle_zz (float): Rotation angle for the IsingZZ gates (default π/8).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        self.ising_angle_xx: float = ising_angle_xx
        self.ising_angle_zz: float = ising_angle_zz

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of the modular block CRot-Ising feature map.

        The 80-dimensional input is split into 10 blocks (of 8 features each).
        1. Hadamard gates create an equal superposition across all qubits.
        2. Each qubit is rotated by an RX gate with angle (π/8) multiplied by the sum of its block's features.
        3. CRot gates are applied between neighboring qubits; the rotation angles for these gates are determined
           from an overlapping subset (last 2 features of current block and first 2 features of the next block), scaled by π/4.
        4. Finally, an entangling layer is applied using IsingXX and IsingZZ gates between adjacent qubits in a ring.

        Args:
            x (np.ndarray): Input data of shape (80,), assumed to be normalized.
        """
        block_size = 8
        overlap_size = 2  # Number of features taken from each block for the overlap

        # Step 1: Apply Hadamard gate to all qubits
        for qubit in range(self.n_qubits):
            qml.Hadamard(wires=[qubit])

        # Step 2: Apply RX rotations on each qubit based on the sum of features in each block
        for qubit in range(self.n_qubits):
            block = x[block_size * qubit : block_size * (qubit + 1)]
            angle_rx = (np.pi / block_size) * np.sum(block)  # Scaling factor π/8
            qml.RX(phi=angle_rx, wires=[qubit])

        # Step 3: Apply CRot gates between neighboring qubits using overlapping features
        # Overlap: last 2 features of current block and first 2 features of the next block
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            block_current = x[block_size * qubit : block_size * (qubit + 1)]
            block_next = x[block_size * next_qubit : block_size * (next_qubit + 1)]
            overlap_features = np.concatenate((block_current[-overlap_size:], block_next[:overlap_size]))
            # The effective rotation angle is computed as (π/|overlap|) times the sum of overlapping features; here |overlap| = 4
            angle_overlap = (np.pi / (2 * overlap_size)) * np.sum(overlap_features)  
            # Apply CRot with the computed angle for all three parameters
            qml.CRot(phi=angle_overlap, theta=angle_overlap, omega=angle_overlap, wires=[qubit, next_qubit])

        # Step 4: Apply an entanglement layer with IsingXX and IsingZZ gates in a ring configuration
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.IsingXX(phi=self.ising_angle_xx, wires=[qubit, next_qubit])
            qml.IsingZZ(phi=self.ising_angle_zz, wires=[qubit, next_qubit])
