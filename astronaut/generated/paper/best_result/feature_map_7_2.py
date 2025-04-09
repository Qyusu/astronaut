import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class RefinedDualStageWithIntermediateCRotOptimizedFeatureMap(BaseFeatureMap):
    """Refined Dual-Stage with Intermediate CRot and Optimized Weighted Averaging Feature Map.
    
    This feature map divides the 80-dimensional input into 5 layers (each with 16 features).
    For each layer l (l = 1,...,5):
      - The first 10 features are encoded on 10 qubits via RY rotations:
            RY(π * x[16*l + j]) for j = 0,...,9.
      - Stage 1 (Immediate neighbor entanglement):
            For each qubit j, a controlled rotation is applied between qubit j and its cyclic neighbor ((j+1) mod 10).
            The rotation angle is computed as π times a weighted sum of two designated features with weights 0.65 and 0.35:
                0.65 * x[16*l + 10 + (j mod 6)] + 0.35 * x[16*l + 10 + ((j+1) mod 6)].
            The gate used is CRZ for odd layers and CRX for even layers.
      - Stage 2 (Next-nearest neighbor entanglement):
            For each qubit j, a CRY gate is applied between qubit j and qubit ((j+2) mod 10).
            The rotation angle is computed as π times a weighted sum of three designated features with weights 0.25, 0.5, and 0.25:
                0.25 * x[16*l + 10 + (j mod 6)] + 0.5 * x[16*l + 10 + ((j+2) mod 6)] + 0.25 * x[16*l + 10 + ((j+4) mod 6)].
      - Intermediate entanglement stage: 
            Fixed qubit pairs (0,5), (1,6), (2,7), (3,8), and (4,9) are entangled via CRot gates.
            The rotation angle for these gates is computed as π times the average over layers of the 12th feature:
                χ = (1/5) * Σₗ x[16*l + 12].
            qml.CRot is used with theta and omega set to 0.
      - Global entanglement: a MultiRZ gate is applied across all qubits with rotation angle
            π * (Σₗ 0.2 * x[16*l + 10]).
    
    Note: For an 80-dimensional input, n_qubits is expected to be 10.
    """
    def __init__(self, n_qubits: int) -> None:
        """Initialize the Refined Dual-Stage with Intermediate CRot and Optimized Weighted Averaging Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits: int = n_qubits
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Refined Dual-Stage with Intermediate CRot and Optimized Weighted Averaging Feature Map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            # Local encoding: apply RY rotations for the first 10 features
            for j in range(self.n_qubits):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            
            # Stage 1: Immediate neighbor entanglement with optimized weighted averaging
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 1) % 6)
                psi1 = 0.65 * x[idx1] + 0.35 * x[idx2]
                angle1 = np.pi * psi1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement with optimized weighted averaging
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                psi2 = 0.25 * x[idx1] + 0.5 * x[idx2] + 0.25 * x[idx3]
                angle2 = np.pi * psi2
                qml.CRY(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
        
        # Intermediate entanglement stage: apply CRot gates for fixed qubit pairs
        # The designated feature is the 12th feature from each layer
        chi = 0.0
        for l in range(5):
            chi += x[16 * l + 12]
        chi = chi / 5.0
        inter_angle = np.pi * chi
        fixed_pairs = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        for a, b in fixed_pairs:
            qml.CRot(phi=inter_angle, theta=0.0, omega=0.0, wires=[a, b])
        
        # Global entanglement: apply a MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += 0.2 * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
