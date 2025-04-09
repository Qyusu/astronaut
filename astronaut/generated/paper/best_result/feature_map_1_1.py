import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class LayeredLocalEncodingRingEntanglement(BaseFeatureMap):
    """Layered Local Encoding with Ring Entanglement feature map.
    
    This feature map partitions the 80-dimensional input data into groups of 8 features per qubit.
    For each qubit j, a series of single-qubit rotations is applied in the following order:
    RX, RY, RZ, RX, RY, RZ, RX, RY with rotation angles given by π times the corresponding
    normalized feature values. After local encoding, a cyclic ring entanglement is introduced
    via CNOT gates between each qubit j and qubit ((j+1) mod n_qubits).
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Layered Local Encoding with Ring Entanglement feature map.
        
        Args:
            n_qubits (int): Number of qubits. For this encoding, n_qubits should ideally be 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits

    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the feature map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (8*n_qubits,). For 10 qubits, shape should be (80,).
        """
        if len(x) != self.n_qubits * 8:
            raise ValueError(f"Input data dimension must be {self.n_qubits * 8}, but got {len(x)}")
        
        # Local encoding: apply a sequence of rotations on each qubit
        for j in range(self.n_qubits):
            base = 8 * j
            qml.RX(phi=np.pi * x[base + 0], wires=j)
            qml.RY(phi=np.pi * x[base + 1], wires=j)
            qml.RZ(phi=np.pi * x[base + 2], wires=j)
            qml.RX(phi=np.pi * x[base + 3], wires=j)
            qml.RY(phi=np.pi * x[base + 4], wires=j)
            qml.RZ(phi=np.pi * x[base + 5], wires=j)
            qml.RX(phi=np.pi * x[base + 6], wires=j)
            qml.RY(phi=np.pi * x[base + 7], wires=j)
        
        # Ring entanglement: apply CNOTs in a cyclic manner
        for j in range(self.n_qubits):
            qml.CNOT(wires=[j, (j + 1) % self.n_qubits])
