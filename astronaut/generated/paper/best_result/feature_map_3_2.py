import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class ScheduledHeterogeneousControlledRotationFeatureMap(BaseFeatureMap):
    """Scheduled Heterogeneous Controlled Rotation Feature Map.
    
    This feature map segments the 80-dimensional input into 5 layers, each comprising 16 features.
    For each layer:
      - The first 10 features are encoded onto 10 qubits via RY rotations using angles π·x.
      - The remaining 6 features form an entanglement block. For each qubit j, a controlled rotation gate is applied
        between qubit j and its neighbor (j+1 modulo the total number of qubits).
        The rotation angle is computed as π times the average of three consecutive features from the entanglement block.
        The type of controlled rotation gate is determined by a round-robin schedule based on j mod 3:
          · If j mod 3 == 0, a CRZ gate is used
          · If j mod 3 == 1, a CRX gate is used
          · If j mod 3 == 2, a CRY gate is used
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Scheduled Heterogeneous Controlled Rotation Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits

    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the feature map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations for the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Entanglement stage with a scheduled heterogeneous gate assignment
            for j in range(self.n_qubits):
                # Compute the average of three consecutive features from the entanglement block
                i1 = base + 10 + (j % 6)
                i2 = base + 10 + ((j + 1) % 6)
                i3 = base + 10 + ((j + 2) % 6)
                ent_angle = np.pi * ((x[i1] + x[i2] + x[i3]) / 3)
                # Select the controlled rotation based on a round-robin schedule using j mod 3
                if j % 3 == 0:
                    qml.CRZ(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
                elif j % 3 == 1:
                    qml.CRX(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRY(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
