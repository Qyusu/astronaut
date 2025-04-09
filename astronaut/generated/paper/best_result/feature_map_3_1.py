import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class GlobalMultiRZEnhancedEntanglementFeatureMap(BaseFeatureMap):
    """Global MultiRZ Enhanced Entanglement Feature Map.
    
    This feature map divides the normalized 80-dimensional input into 5 layers, each comprising 16 features.
    For each layer l (l = 1,...,5):
      - The first 10 features are encoded on 10 qubits via RY rotations with angles π·x,
        preserving local feature information.
      - The remaining 6 features are used in a local entanglement stage. Each qubit is paired with its neighbor
        and subjected to a controlled rotation gate whose rotation angle is computed as π times the average of
        three consecutive features from the entanglement block. The gate type alternates depending on the layer:
          · CRZ is applied for odd-numbered layers
          · CRX is applied for even-numbered layers
    After processing all layers, a global entanglement operation is performed using a MultiRZ gate applied to all qubits.
    The MultiRZ rotation angle is set by π times the sum over layers of the average of three predetermined features
    (taken from positions 10, 11, and 12 of each layer) in the entanglement block.

    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Global MultiRZ Enhanced Entanglement Feature Map.
        
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
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations using the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Local entanglement stage: apply controlled rotation between each qubit and its neighbor
            for j in range(self.n_qubits):
                # Determine three consecutive indices (cyclic within the 6 entanglement features)
                i1 = base + 10 + (j % 6)
                i2 = base + 10 + ((j + 1) % 6)
                i3 = base + 10 + ((j + 2) % 6)
                ent_angle = np.pi * ((x[i1] + x[i2] + x[i3]) / 3)
                # Use CRZ for odd layers, CRX for even layers (layer index l+1 is considered)
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=ent_angle, wires=[j, (j + 1) % self.n_qubits])
        
        # Global entanglement: apply a MultiRZ gate across all qubits
        global_angle = 0.0
        for l in range(5):
            base = 16 * l
            # Use predetermined indices from the entanglement block: positions 10, 11, and 12
            layer_avg = (x[base + 10] + x[base + 11] + x[base + 12]) / 3
            global_angle += layer_avg
        global_angle *= np.pi
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
