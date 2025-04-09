import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# New imports can be added below this line if needed.

class OptimizedStandardizedEntanglementFeatureMapV6(BaseFeatureMap):
    """Optimized Standardized Entanglement Feature Map v6.
    
    This feature map preserves the full 80-dimensional PCA-reduced input by evenly partitioning it among 10 qubits,
    where each qubit i (0 ≤ i ≤ 9) receives features f_{i,j} = x[8*i + j] for j = 0,...,7.
    The circuit applies the following operations:
      - U_local: A local rotation block is applied to each qubit with a variable cyclic order. For even-indexed
                 qubits the order is [RY, RX, RZ, RY] and for odd-indexed qubits the order is [RX, RZ, RY, RX],
                 encoding features f_{i,1}–f_{i,4} (indices 0–3) scaled by π.
      - Nearest-neighbor entanglement: A ring of ControlledPhaseShift (CP) gates is applied between adjacent qubits.
      - Intermediate entanglement: CRZ gates connect qubit i with qubit (i+3) mod 10 with rotation angles given by
                 η · π*(2*f_{i,5} + 3*f_{(i+3),6})/5, where η is a calibration factor, f_{i,5} is from index 4 and
                 f_{(i+3),6} is from index 5 of the partner qubit.
      - Global entanglement: A MultiRZ gate acts on all qubits with a rotation angle ζ · π*(Σ_{i=0}^{9}(f_{i,7}+f_{i,8}))/20,
                 where ζ is a calibration factor and f_{i,7} and f_{i,8} are from indices 6 and 7 respectively.
    
    The overall circuit implements:
      |Φ(x)⟩ = (∏_{i=0}^{9} U^{(i)}_{local}) · (∏_{i=0}^{9} CP(q_i, q_{(i+1) mod 10})) ·
              (∏_{i=0}^{9} CRZ_{i,(i+3) mod 10}(η · π*(2f_{i,5}+3f_{(i+3),6})/5)) ·
              MultiRZ(ζ · π*(Σ_{i=0}^{9}(f_{i,7}+f_{i,8}))/20) |0⟩.
    
    This design maintains a low circuit depth while capturing rich inter-feature correlations and
    includes calibration factors (η and ζ) to mitigate hardware noise effects.
    """
    
    def __init__(self, n_qubits: int, cp_angle: float = np.pi/4, eta: float = 1.0, zeta: float = 1.0) -> None:
        """Initialize the Optimized Standardized Entanglement Feature Map v6.
        
        Args:
            n_qubits (int): Number of qubits (expected to be 10 for an 80-dimensional input).
            cp_angle (float): Fixed phase angle for the ControlledPhaseShift gates (default π/4).
            eta (float): Calibration factor for the intermediate entanglement layer (default 1.0).
            zeta (float): Calibration factor for the global entanglement layer (default 1.0).
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        self.n_qubits = n_qubits
        self.cp_angle = cp_angle
        self.eta = eta
        self.zeta = zeta
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create the quantum circuit for the Optimized Standardized Entanglement Feature Map v6.
        
        Args:
            x (np.ndarray): Input feature vector of shape (80,), with features normalized to [0, 1].
        """
        # Step 1: U_local - Local rotation block with a variable cyclic order
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f1 = x[base_idx + 0]
            f2 = x[base_idx + 1]
            f3 = x[base_idx + 2]
            f4 = x[base_idx + 3]
            
            # Use different cyclic orders based on whether the qubit index is even or odd
            if qubit % 2 == 0:
                local_order = [qml.RY, qml.RX, qml.RZ, qml.RY]
            else:
                local_order = [qml.RX, qml.RZ, qml.RY, qml.RX]
            
            angles = [np.pi * f1, np.pi * f2, np.pi * f3, np.pi * f4]
            for gate, angle in zip(local_order, angles):
                gate(phi=angle, wires=[qubit])
        
        # Step 2: Nearest-neighbor entanglement using ControlledPhaseShift (CP) gates
        for qubit in range(self.n_qubits):
            next_qubit = (qubit + 1) % self.n_qubits
            qml.ControlledPhaseShift(phi=self.cp_angle, wires=[qubit, next_qubit])
        
        # Step 3: Intermediate entanglement using CRZ gates between qubit i and (i+3) mod n_qubits
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f5 = x[base_idx + 4]
            partner = (qubit + 3) % self.n_qubits
            # f6 is the 6th feature (index 5) of the partner qubit
            f6_partner = x[8 * partner + 5]
            angle_crz = self.eta * np.pi * (2 * f5 + 3 * f6_partner) / 5.0
            qml.CRZ(phi=angle_crz, wires=[qubit, partner])
        
        # Step 4: Global entanglement using MultiRZ across all qubits
        total = 0.0
        for qubit in range(self.n_qubits):
            base_idx = 8 * qubit
            f7 = x[base_idx + 6]
            f8 = x[base_idx + 7]
            total += (f7 + f8)
        global_angle = self.zeta * np.pi * (total / 20.0)
        qml.MultiRZ(theta=global_angle, wires=list(range(self.n_qubits)))
