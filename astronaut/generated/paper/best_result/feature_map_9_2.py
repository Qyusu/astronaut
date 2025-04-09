import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# new imports can be added below this line if needed.


class AuxiliaryRegisterRedundantMultiScaleEntanglementFeatureMap(BaseFeatureMap):
    """Auxiliary Register Redundant Multi-Scale Entanglement Feature Map.
    
    This feature map partitions the 10-qubit system into two registers of 5 qubits each:
      - Main register: qubits 0–4
      - Auxiliary register: qubits 5–9
    The 80-dimensional PCA-reduced input is divided into 5 layers (each with 16 features).
    For each layer l (l = 0,...,4):
      - Main register encoding: The first 5 features are embedded via RY rotations onto qubits 0–4:
            RY(π * x[16*l + j]) for j = 0,...,4.
      - Auxiliary register encoding: The next 5 features are embedded via RY rotations onto qubits 5–9:
            RY(π * x[16*l + 5 + j]) for j = 0,...,4.
      - Intra-layer entanglement is performed separately:
          * For the main register (qubits 0–4):
              Nearest-neighbor entanglement is applied using controlled rotations.
              The rotation angle is computed as π times a weighted pairwise average:
                  α^(l)_j = w_main[l][j][0] * x[16*l + 10 + (j mod 6)] + w_main[l][j][1] * x[16*l + 10 + ((j+1) mod 6)].
              CRZ is applied for layers where (l mod 2 == 0) and CRX for layers where (l mod 2 == 1), on qubits j and (j+1 mod 5).
          * For the auxiliary register (qubits 5–9):
              Next-nearest neighbor entanglement is applied using ControlledPhaseShift gates.
              The rotation angle is computed as π times a weighted triple average:
                  β^(l)_j = v_aux[l][j][0] * x[16*l + 10 + (j mod 6)] + v_aux[l][j][1] * x[16*l + 10 + ((j+2) mod 6)] + v_aux[l][j][2] * x[16*l + 10 + ((j+4) mod 6)].
          * Cross-register entanglement:
              For each qubit j in the main register, a CRot gate couples it with the corresponding qubit (j+5) in the auxiliary register.
              The rotation angle is computed as π times a weighted average of two designated entanglement features; here,
              we use γ^(l)_j = 0.5*(x[16*l + 12] + x[16*l + 13]).
      - Global entanglement: A MultiRZ gate is applied across all qubits with rotation angle
              π * (Σₗ δₗ * x[16*l + 10]),
          where δₗ are non-uniform weights (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
    
    The additional parameters are:
      - w_main: a 5x5x2 tensor of weights for main register entanglement (default: all 0.5)
      - v_aux: a 5x5x3 tensor of weights for auxiliary register entanglement (default: all 1/3)
      - delta_weights: list of 5 non-uniform weights for global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10])
    
    Note: The input x is expected to have shape (80,).
    """
    def __init__(self, n_qubits: int, 
                 w_main: list = None, 
                 v_aux: list = None, 
                 delta_weights: list = None) -> None:
        """Initialize the Auxiliary Register Redundant Multi-Scale Entanglement Feature Map.
        
        Args:
            n_qubits (int): Number of qubits, ideally 10 (with registers partitioned as 5 and 5).
            w_main (list): 5x5x2 tensor of weights for main register entanglement (default: 0.5 for all entries).
            v_aux (list): 5x5x3 tensor of weights for auxiliary register entanglement (default: 1/3 for all entries).
            delta_weights (list): List of 5 non-uniform weights for global entanglement (default: [0.15, 0.25, 0.35, 0.15, 0.10]).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        if w_main is None:
            # For 5 layers, 5 qubits in main register, each with 2 weights
            self.w_main = [[[0.5, 0.5] for _ in range(5)] for _ in range(5)]
        else:
            self.w_main = w_main
        if v_aux is None:
            # For 5 layers, 5 qubits in auxiliary register, each with 3 weights
            self.v_aux = [[[1/3, 1/3, 1/3] for _ in range(5)] for _ in range(5)]
        else:
            self.v_aux = v_aux
        if delta_weights is None:
            self.delta_weights = [0.15, 0.25, 0.35, 0.15, 0.10]
        else:
            self.delta_weights = delta_weights
        
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Auxiliary Register Redundant Multi-Scale Entanglement Feature Map.
        
        Args:
            x (np.ndarray): Input data, expected to have shape (80,), corresponding to 5 layers of 16 features each.
        """
        expected_length = 5 * 16
        if len(x) != expected_length:
            raise ValueError(f"Input data dimension must be {expected_length}, but got {len(x)}")
        
        # Process each of the 5 layers
        for l in range(5):
            base = 16 * l
            # Main register encoding: apply RY rotations for first 5 features onto qubits 0-4
            for j in range(5):
                angle = np.pi * x[base + j]
                qml.RY(phi=angle, wires=j)
            # Auxiliary register encoding: apply RY rotations for next 5 features onto qubits 5-9
            for j in range(5):
                angle = np.pi * x[base + 5 + j]
                qml.RY(phi=angle, wires=j + 5)
            
            # Intra-layer entanglement for main register (qubits 0-4): nearest-neighbor entanglement
            for j in range(5):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 1) % 6)
                alpha = self.w_main[l][j][0] * x[idx_a] + self.w_main[l][j][1] * x[idx_b]
                angle_main = np.pi * alpha
                if (l % 2) == 0:
                    qml.CRZ(phi=angle_main, wires=[j, (j + 1) % 5])
                else:
                    qml.CRX(phi=angle_main, wires=[j, (j + 1) % 5])
            
            # Intra-layer entanglement for auxiliary register (qubits 5-9): next-nearest neighbor entanglement
            for j in range(5):
                idx_a = base + 10 + (j % 6)
                idx_b = base + 10 + ((j + 2) % 6)
                idx_c = base + 10 + ((j + 4) % 6)
                beta_val = (self.v_aux[l][j][0] * x[idx_a] + 
                            self.v_aux[l][j][1] * x[idx_b] + 
                            self.v_aux[l][j][2] * x[idx_c])
                angle_aux = np.pi * beta_val
                # For auxiliary register, use ControlledPhaseShift between qubit (j+5) and its next-nearest neighbor ((j+2)%5 + 5)
                qml.ControlledPhaseShift(phi=angle_aux, wires=[j + 5, ((j + 2) % 5) + 5])
            
            # Cross-register entanglement: couple each main register qubit with its corresponding auxiliary qubit
            for j in range(5):
                # Here, we compute a weighted average of two designated entanglement features,
                # e.g., using features at indices base+12 and base+13
                gamma = 0.5 * (x[base + 12] + x[base + 13])
                angle_cross = np.pi * gamma
                qml.CRot(phi=angle_cross, theta=0.0, omega=0.0, wires=[j, j + 5])
        
        # Global entanglement: apply a MultiRZ gate across all 10 qubits
        global_sum = 0.0
        for l in range(5):
            global_sum += self.delta_weights[l] * x[16 * l + 10]
        global_angle = np.pi * global_sum
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
