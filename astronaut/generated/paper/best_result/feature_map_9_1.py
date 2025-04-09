import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class EnhancedQubitSpecificMultiStageFeatureMap(BaseFeatureMap):
    """Enhanced Qubit-Specific Multi-Stage Feature Map with Mid-Range Entanglement.
    
    This feature map partitions the normalized 80-dimensional PCA-reduced input into 5 layers (each with 16 features).
    For each layer l (l = 0,...,4):
      - Local encoding: The first 10 features are embedded onto a 10-qubit register via RY rotations:
            RY(π * x[16*l + j]) for j = 0,...,9.
      - Stage 1 (Immediate neighbor entanglement):
            For each qubit j, a controlled rotation is applied between qubit j and its cyclic neighbor ((j+1) mod 10).
            The rotation angle is computed as π times a weighted pairwise average of two designated entanglement features:
                θ₁ = w1[l,j] * x[16*l + 10 + (j mod 6)] + (1 - w1[l,j]) * x[16*l + 10 + ((j+1) mod 6)].
            CRZ is applied if the layer index is even (i.e. l+1 odd) and CRX if odd.
      - Stage 2 (Next-nearest neighbor entanglement):
            For each qubit j, a ControlledPhaseShift gate couples qubit j with qubit ((j+2) mod 10).
            The rotation angle is computed as π times a weighted triple average:
                θ₂ = v[l,j,0] * x[16*l + 10 + (j mod 6)] + v[l,j,1] * x[16*l + 10 + ((j+2) mod 6)] + v[l,j,2] * x[16*l + 10 + ((j+4) mod 6)].
      - Stage 3 (Mid-range entanglement):
            For each qubit j, a ControlledPhaseShift gate couples qubit j with qubit ((j+3) mod 10).
            The rotation angle is computed as π times a weighted pairwise average:
                θ₃ = u1[l,j] * x[16*l + 10 + (j mod 6)] + (1 - u1[l,j]) * x[16*l + 10 + ((j+1) mod 6)].
      - After processing all layers, an intermediate cross-layer entanglement stage is applied:
            For each qubit j, a CRot gate entangles qubit j with qubit ((j+5) mod 10) with rotation angle
                φ_j = π * (1/5)* Σₗ (cross_weights[l,j] * x[16*l + 12]).
      - Global entanglement: A MultiRZ gate is applied across all qubits with rotation angle
                π * (λ * (Σₗ δₗ * x[16*l + 10] + β)),
            where λ is an offline-calibrated noise scaling factor, β is an offline bias compensation offset, and δₗ are non-uniform weights (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
    
    The additional parameters (offline calibrated weights) are:
      - w1: a 5x10 matrix of weights for Stage 1 (default: all 0.5)
      - v: a 5x10x3 tensor of weights for Stage 2 (default: all 1/3)
      - u1: a 5x10 matrix of weights for Stage 3 (default: all 0.5)
      - cross_weights: a 5x10 matrix for the cross-layer entanglement stage (default: all 1.0)
      - lambda_factor: noise scaling factor (default: 1.0)
      - beta: bias compensation offset (default: 0.0)
      - delta_weights: list of 5 non-uniform weights for global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10])
    
    Note: The input x is expected to have shape (80,). The qubit register size is 10.
    """
    def __init__(self, n_qubits: int, 
                 w1: list = None, 
                 v: list = None, 
                 u1: list = None, 
                 cross_weights: list = None, 
                 lambda_factor: float = 1.0,
                 beta: float = 0.0,
                 delta_weights: list = None) -> None:
        """Initialize the Enhanced Qubit-Specific Multi-Stage Feature Map with Mid-Range Entanglement.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10.
            w1 (list): 5x10 weight matrix for Stage 1 (default: 0.5 for all entries).
            v (list): 5x10x3 weight tensor for Stage 2 (default: 1/3 for all entries).
            u1 (list): 5x10 weight matrix for Stage 3 (default: 0.5 for all entries).
            cross_weights (list): 5x10 weight matrix for cross-layer entanglement (default: 1.0 for all entries).
            lambda_factor (float): Noise scaling factor for global entanglement (default: 1.0).
            beta (float): Bias compensation offset for global entanglement (default: 0.0).
            delta_weights (list): List of 5 non-uniform weights for global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        # Set default weights if not provided
        if w1 is None:
            self.w1 = [[0.5 for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.w1 = w1
        if v is None:
            self.v = [[[1/3 for _ in range(3)] for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.v = v
        if u1 is None:
            self.u1 = [[0.5 for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.u1 = u1
        if cross_weights is None:
            self.cross_weights = [[1.0 for _ in range(n_qubits)] for _ in range(5)]
        else:
            self.cross_weights = cross_weights
        self.lambda_factor = lambda_factor
        self.beta = beta
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Enhanced Qubit-Specific Multi-Stage Feature Map with Mid-Range Entanglement.
        
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
            
            # Stage 1: Immediate neighbor entanglement with qubit-specific weighted pair average
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                theta1 = self.w1[l][j] * x[idx_a] + (1 - self.w1[l][j]) * x[idx_b]
                angle1 = np.pi * theta1
                # Use CRZ if layer index is even (l+1 odd), CRX otherwise
                if (l % 2) == 0:
                    qml.CRZ(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
                else:
                    qml.CRX(phi=angle1, wires=[j, (j + 1) % self.n_qubits])
            
            # Stage 2: Next-nearest neighbor entanglement via ControlledPhaseShift gates
            for j in range(self.n_qubits):
                idx1 = base + 10 + (j % 6)
                idx2 = base + 10 + ((j + 2) % 6)
                idx3 = base + 10 + ((j + 4) % 6)
                theta2 = (self.v[l][j][0] * x[idx1] + 
                          self.v[l][j][1] * x[idx2] + 
                          self.v[l][j][2] * x[idx3])
                angle2 = np.pi * theta2
                qml.ControlledPhaseShift(phi=angle2, wires=[j, (j + 2) % self.n_qubits])
            
            # Stage 3: Mid-range entanglement between qubits separated by 3
            for j in range(self.n_qubits):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                theta3 = self.u1[l][j] * x[idx_a] + (1 - self.u1[l][j]) * x[idx_b]
                angle3 = np.pi * theta3
                qml.ControlledPhaseShift(phi=angle3, wires=[j, (j + 3) % self.n_qubits])
        
        # Intermediate cross-layer entanglement stage: couple qubit j with qubit ((j+5) mod n_qubits)
        for j in range(self.n_qubits):
            accum = 0.0
            for l in range(5):
                accum += self.cross_weights[l][j] * x[16 * l + 12]
            phi_j = np.pi * (accum / 5.0)
            qml.CRot(phi=phi_j, theta=0.0, omega=0.0, wires=[j, (j + 5) % self.n_qubits])
        
        # Global entanglement: MultiRZ gate across all qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * self.lambda_factor * (global_sum + self.beta)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
