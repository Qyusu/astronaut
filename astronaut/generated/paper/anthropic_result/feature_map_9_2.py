import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap


class RefinedGlobalEntanglementFeatureMap(BaseFeatureMap):
    """Refined Global Entanglement with Balanced Gate Distribution feature map.

    This feature map implements a balanced rotation gate distribution
    with refined global entanglement layer and information-theoretic
    phase shifts. The implementation includes a strategic CZ triplet pattern
    and a symmetry-preserving alternating Hadamard pattern.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
        >>> feature_map = RefinedGlobalEntanglementFeatureMap(n_qubits=10)
    """

    def __init__(
        self, 
        n_qubits: int, 
        scale_factor: float = np.pi, 
        offset: float = np.pi/4,
        phase1: float = np.pi/4,
        phase2: float = 2*np.pi/5,
        phase3: float = 3*np.pi/7,
        final_phase1: float = np.pi/5,
        final_phase2: float = np.pi/8,
        reps: int = 2
    ) -> None:
        """Initialize the Refined Global Entanglement Feature Map.

        Args:
            n_qubits (int): number of qubits
            scale_factor (float, optional): Scaling factor for feature angles. Defaults to np.pi.
            offset (float, optional): Offset for feature angles. Defaults to np.pi/4.
            phase1 (float, optional): Phase shift after local entanglement. Defaults to np.pi/4.
            phase2 (float, optional): Phase shift after medium entanglement. Defaults to 2*np.pi/5.
            phase3 (float, optional): Phase shift after global entanglement. Defaults to 3*np.pi/7.
            final_phase1 (float, optional): First phase shift in final Hadamard pattern. Defaults to np.pi/5.
            final_phase2 (float, optional): Second phase shift in final Hadamard pattern. Defaults to np.pi/8.
            reps (int, optional): Number of repetitions. Defaults to 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.scale_factor: float = scale_factor
        self.offset: float = offset
        self.phase1: float = phase1
        self.phase2: float = phase2
        self.phase3: float = phase3
        self.final_phase1: float = final_phase1
        self.final_phase2: float = final_phase2
        self.reps: int = reps
        
        # Define global connection distance
        self.global_distance = max(1, n_qubits // 4)
        
        # Define triplets for controlled-Z gates
        self.cz_triplets = [
            (0, 3, 6), (1, 4, 7), (2, 5, 8), (3, 6, 9), (4, 7, 0),
            (5, 8, 1), (6, 9, 2), (7, 0, 3), (8, 1, 4), (9, 2, 5)
        ]

    def _encode_features(self, x: np.ndarray) -> None:
        """Apply feature encoding with balanced gate distribution.
        
        Args:
            x (np.ndarray): Input data of shape (80,)
        """
        # Ensure we don't go beyond the available features
        n_features = min(80, len(x))
        
        # Features 1-10 → Rx rotations on qubits 1-10
        for i in range(min(10, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=i % self.n_qubits)
        
        # Features 11-20 → Ry rotations on qubits 1-10
        for i in range(10, min(20, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 10) % self.n_qubits)
        
        # Features 21-30 → Ry rotations on qubits 1-10 (second Ry layer)
        for i in range(20, min(30, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 20) % self.n_qubits)
        
        # Features 31-40 → Rz rotations on qubits 1-10
        for i in range(30, min(40, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 30) % self.n_qubits)
        
        # Features 41-50 → Rx rotations on qubits 1-10 (second round)
        for i in range(40, min(50, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RX(phi=angle, wires=(i - 40) % self.n_qubits)
        
        # Features 51-60 → Ry rotations on qubits 1-10 (third Ry layer)
        for i in range(50, min(60, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 50) % self.n_qubits)
        
        # Features 61-70 → Ry rotations on qubits 1-10 (fourth Ry layer)
        for i in range(60, min(70, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RY(phi=angle, wires=(i - 60) % self.n_qubits)
        
        # Features 71-80 → Rz rotations on qubits 1-10 (second Rz layer)
        for i in range(70, min(80, n_features)):
            angle = self.scale_factor * x[i] + self.offset
            qml.RZ(phi=angle, wires=(i - 70) % self.n_qubits)

    def _apply_local_entanglement(self) -> None:
        """Apply CNOT gates between adjacent qubits (Layer 1)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
    
    def _apply_medium_entanglement(self) -> None:
        """Apply CNOT gates between qubits separated by distance 2 (Layer 2)."""
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
    
    def _apply_global_entanglement(self) -> None:
        """Apply CNOT gates with refined global entanglement pattern.
        
        For a 10-qubit system, this connects qubits (i → i+2) and (i → i+5) alternately.
        """
        for i in range(self.n_qubits):
            if i % 2 == 0:  # Even-indexed qubits
                qml.CNOT(wires=[i, (i + 2) % self.n_qubits])
            else:  # Odd-indexed qubits
                qml.CNOT(wires=[i, (i + 5) % self.n_qubits])
    
    def _apply_phase1(self) -> None:
        """Apply phase shift after Layer 1: Rz(π/4) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase1, wires=i)
    
    def _apply_phase2(self) -> None:
        """Apply phase shift after Layer 2: Rz(2π/5) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase2, wires=i)
    
    def _apply_phase3(self) -> None:
        """Apply phase shift after Layer 3: Rz(3π/7) to all qubits."""
        for i in range(self.n_qubits):
            qml.RZ(phi=self.phase3, wires=i)
    
    def _apply_cz_triplets(self) -> None:
        """Apply controlled-Z gates to strategic triplets."""
        for a, b, c in self.cz_triplets:
            if a < self.n_qubits and b < self.n_qubits and c < self.n_qubits:
                qml.CZ(wires=[a, b])
                qml.CZ(wires=[b, c])
                qml.CZ(wires=[c, a])
    
    def _apply_symmetry_preserving_hadamard(self) -> None:
        """Apply symmetry-preserving alternating Hadamard pattern."""
        # Apply H gates to odd-indexed qubits
        for i in range(1, self.n_qubits, 2):
            qml.Hadamard(wires=i)
        
        # Apply Rz(π/5) to all qubits
        for i in range(self.n_qubits):
            qml.RZ(phi=self.final_phase1, wires=i)
        
        # Apply H gates to even-indexed qubits
        for i in range(0, self.n_qubits, 2):
            qml.Hadamard(wires=i)
        
        # Apply a final Rz(π/8) to all qubits
        for i in range(self.n_qubits):
            qml.RZ(phi=self.final_phase2, wires=i)
    
    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.
        
        Args:
            x (np.ndarray): input data shape is (80,).
        """
        # Hybrid repetition structure with specified number of repetitions
        for _ in range(self.reps):
            # Encode features with balanced gate distribution
            self._encode_features(x)
            
            # Apply three-layer entanglement with information-theoretic phase shifts
            self._apply_local_entanglement()
            self._apply_phase1()
            
            self._apply_medium_entanglement()
            self._apply_phase2()
            
            self._apply_global_entanglement()
            self._apply_phase3()
            
            # Apply strategic controlled-Z triplet pattern
            self._apply_cz_triplets()
        
        # Additional feature encoding layer
        self._encode_features(x)
        
        # Apply symmetry-preserving alternating Hadamard pattern
        self._apply_symmetry_preserving_hadamard()