import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class EnhancedNonUniformWeightedTripleStageFeatureMap(BaseFeatureMap):
    """Enhanced Non-Uniform Weighted Triple-Stage Feature Map with Cross-Layer Coupling and Offline Error Mitigation.
    
    This feature map partitions the 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 0,...,4):
      - Local encoding: The first 10 features are embedded on 10 qubits via RY rotations, i.e.,
            RY(π * x[16*l + j]) for j = 0,...,9.
      - Stage 1 (Immediate neighbor entanglement):
            For each qubit j, a controlled rotation is applied between qubit j and its cyclic neighbor ((j+1) mod 10).
            The rotation angle is computed as π times a weighted average using non-uniform weights:
                θ₁ = 0.7 * x[16*l + 10 + (j mod 6)] + 0.3 * x[16*l + 10 + ((j+1) mod 6)].
            CRZ is used if the layer index is even (i.e. l+1 odd) and CRX if odd.
      - Stage 2 (Next-nearest neighbor entanglement):
            For each qubit j, a ControlledPhaseShift gate couples qubit j with qubit ((j+2) mod 10).
            The rotation angle is computed as π times a weighted triple average:
                θ₂ = 0.2 * x[16*l + 10 + (j mod 6)] + 0.5 * x[16*l + 10 + ((j+2) mod 6)] + 0.3 * x[16*l + 10 + ((j+4) mod 6)].
      - After all layers, an intermediate cross-layer entanglement stage is applied:
            For each qubit j, a CRot gate entangles qubit j with qubit ((j+5) mod 10) with rotation angle
                φ'_j = β_j * (1/5 * Σₗ x[16*l + 12]),
            where β_j are offline-calibrated factors (default is 1 for all qubits).
      - Global entanglement: A MultiRZ gate is applied across all qubits with rotation angle
                π * (β_global * Σₗ δₗ * x[16*l + 10]),
            where δₗ are non-uniform weights (default: [0.15, 0.25, 0.35, 0.15, 0.10]) and β_global is an offline calibration factor.
            
    Note: The input x is expected to have shape (80,). The qubit register size should be 10.
    """
    def __init__(self, n_qubits: int, beta_factors: list = None, beta_global: float = 1.0, delta_weights: list = None) -> None:
        """Initialize the Enhanced Non-Uniform Weighted Triple-Stage Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            beta_factors (list): Offline calibrated factors for cross-layer entanglement (default: [1.0]*n_qubits).
            beta_global (float): Global calibration factor (default: 1.0).
            delta_weights (list): Non-uniform weights for each layer in global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        if beta_factors is None:
            self.beta_factors = [1.0 for _ in range(n_qubits)]
        else:
            self.beta_factors = beta_factors
        self.beta_global = beta_global
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Enhanced Non-Uniform Weighted Triple-Stage Feature Map.
        
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
            
            # Stage 1: Immediate neighbor entanglement using controlled rotations
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 1) % 6)
                theta1 = 0.7 * x[idx1] + 0.3 * x[idx2]
                angle1 = np.pi * theta1
                if (l + 1) % 2 == 1:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement using ControlledPhaseShift gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                theta2 = 0.2 * x[idx1] + 0.5 * x[idx2] + 0.3 * x[idx3]
                angle2 = np.pi * theta2
                qml.ControlledPhaseShift(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
        
        # Intermediate cross-layer entanglement stage: apply CRot gates across qubits j and (j+5) mod n_qubits
        chi = 0.0
        for l in range(5):
            chi += x[16 * l + 12]
        chi /= 5.0
        for j in range(self.n_qubits):
            cross_angle = np.pi * (self.beta_factors[j] * chi)
            target = (j + 5) % self.n_qubits
            qml.CRot(phi=cross_angle, theta=0.0, omega=0.0, wires=[j, target])
        
        # Global entanglement: apply a MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * (self.beta_global * global_sum)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
