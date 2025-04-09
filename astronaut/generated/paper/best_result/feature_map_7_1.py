import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class TripleStageIntraAndInterLayerEntanglementFeatureMap(BaseFeatureMap):
    """Triple-Stage Intra- and Inter-Layer Entanglement Feature Map.
    
    This feature map divides the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - The first 10 features are encoded locally on 10 qubits using RY rotations, i.e.,
          RY(π * x[16*l + j]) for j=0,...,9.
      - Stage 1 (Intra-layer immediate neighbor entanglement):
          For each qubit j, a controlled rotation is applied between qubit j and its cyclic neighbor ((j+1) mod 10).
          The rotation angle is computed as π times a weighted sum of two designated features:
              0.6 * x[16*l + 10 + (j mod 6)] + 0.4 * x[16*l + 10 + ((j+1) mod 6)].
          The controlled rotation gate used is CRZ if the layer is odd and CRX if the layer is even.
      - Stage 2 (Intra-layer next-nearest neighbor entanglement):
          For each qubit j, a CRY gate is applied between qubit j and qubit ((j+2) mod 10).
          The rotation angle is computed as π times a weighted sum of three designated features:
              0.3 * x[16*l + 10 + (j mod 6)] + 0.4 * x[16*l + 10 + ((j+2) mod 6)] + 0.3 * x[16*l + 10 + ((j+4) mod 6)].
    After processing all layers, an intermediate inter-layer entanglement stage is introduced: 
      - For each qubit j, a CRot gate is applied between qubit j and qubit ((j+5) mod 10).
        The rotation angle for these gates is computed as π times the average over layers of the 15th feature, i.e.
            φ = (1/5) * Σₗ x[16*l + 15].
        Here, qml.CRot is used with fixed theta and omega set to 0.
    Finally, a global MultiRZ gate is applied across all 10 qubits with rotation angle
          π * (Σₗ 0.2 * x[16*l + 10]),
    capturing long-range inter-layer correlations.

    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Triple-Stage Intra- and Inter-Layer Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Triple-Stage Intra- and Inter-Layer Entanglement Feature Map.
        
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
            
            # Stage 1: Immediate neighbor entanglement (CRZ for odd layers, CRX for even layers)
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 1) % 6)
                theta1 = 0.6 * x[idx1] + 0.4 * x[idx2]
                angle1 = np.pi * theta1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement using CRY gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                theta2 = 0.3 * x[idx1] + 0.4 * x[idx2] + 0.3 * x[idx3]
                angle2 = np.pi * theta2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
        
        # Intermediate inter-layer entanglement stage: apply CRot gates between qubits j and (j+5) mod 10
        # Compute the average of the 15th feature from each layer
        chi = 0.0
        for l in range(5):
            chi += x[16 * l + 15]
        chi = chi / 5.0
        inter_angle = np.pi * chi
        for j in range(self.n_qubits):
            target = (j + 5) % self.n_qubits
            # Using CRot with theta and omega set to 0
            qml.CRot(phi=inter_angle, theta=0.0, omega=0.0, wires=[j, target])
        
        # Final global entanglement: apply a MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += 0.2 * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
