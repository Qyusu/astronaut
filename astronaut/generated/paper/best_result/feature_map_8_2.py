import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class AlternatingCPSIntraLayerWithCrossLayerCRotGlobalFeatureMap(BaseFeatureMap):
    """Alternating ControlledPhaseShift Intra-Layer with Cross-Layer CRot Global Feature Map.
    
    This feature map partitions the 80-dimensional normalized input into 5 layers (each with 16 features).
    For each layer l (l = 0,...,4):
      - Local encoding: The first 10 normalized features are embedded on 10 qubits via RY rotations:
              RY(π * Norm(x[16*l + j])) for j = 0,...,9, where Norm(·) denotes a fixed normalization.
      - Stage 1 (Immediate neighbor entanglement):
              For each qubit j, a controlled rotation is applied between qubit j and its neighbor ((j+1) mod 10).
              The rotation angle is computed as π times a weighted average:
                  ψ₁ = 0.68 * Norm(x[16*l + 10 + (j mod 6)]) + 0.32 * Norm(x[16*l + 10 + ((j+1) mod 6)]).
              CRZ is used for odd layers and CRX for even layers.
      - Stage 2 (Next-nearest neighbor entanglement):
              For each qubit j, a ControlledPhaseShift (CPS) gate couples qubit j with qubit ((j+2) mod 10).
              The rotation angle is computed as π times a weighted triple average:
                  ψ₂ = 0.25 * Norm(x[16*l + 10 + (j mod 6)]) + 0.5 * Norm(x[16*l + 10 + ((j+2) mod 6)]) + 0.25 * Norm(x[16*l + 10 + ((j+4) mod 6)]).
      - Intermediate cross-layer entanglement:
              Fixed qubit pairs (0,5), (1,6), (2,7), (3,8), (4,9) are entangled via CRot gates with rotation angle
                  χ = (1/5) * Σₗ Norm(x[16*l + 12]).
      - Global entanglement:
              A MultiRZ gate is applied across all qubits with rotation angle
                  π * (Σₗ δₗ * Norm(x[16*l + 10])),
              where δₗ are predetermined non-uniform weights (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
    
    Note: The normalization function Norm(·) is assumed to be the identity if inputs are already normalized.
    """
    def __init__(self, n_qubits: int, delta_weights: list = None) -> None:
        """Initialize the Alternating CPS Intra-Layer with Cross-Layer CRot Global Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            delta_weights (list): Non-uniform weights for global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Alternating CPS Intra-Layer with Cross-Layer CRot Global Feature Map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Define a normalization function. Here we assume inputs are already normalized.
        norm = lambda a: a
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations on the first 10 normalized features
            for j in range(self.n_qubits):
                angle = np.pi * norm(x[base + j])
                qml.RY(phi=angle, wires=j)
            
            # Stage 1: Immediate neighbor entanglement with alternating controlled rotations
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 1) % 6)
                psi1 = 0.68 * norm(x[idx1]) + 0.32 * norm(x[idx2])
                angle1 = np.pi * psi1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement using ControlledPhaseShift gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                psi2 = 0.25 * norm(x[idx1]) + 0.5 * norm(x[idx2]) + 0.25 * norm(x[idx3])
                angle2 = np.pi * psi2
                qml.ControlledPhaseShift(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
        
        # Intermediate cross-layer entanglement: fixed qubit pairs
        chi = 0.0
        for l in range(5):
            chi += norm(x[16 * l + 12])
        chi /= 5.0
        fixed_pairs = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        for a, b in fixed_pairs:
            qml.CRot(phi=np.pi * chi, theta=0.0, omega=0.0, wires=[a, b])
        
        # Global entanglement: apply a MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * norm(x[16 * l + 10])
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
