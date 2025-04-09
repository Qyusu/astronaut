import numpy as np
import pennylane as qml
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class LEFMFGFeatureMap(BaseFeatureMap):
    """LEFM with Feature Grouping (LEFM-FG) class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int, group_size: int = 2) -> None:
        """Initialize the LEFM with Feature Grouping feature map class.

        Args:
            n_qubits (int): number of qubits
            group_size (int): Size of the feature groups. Default is 2.
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits
        self.group_size: int = group_size

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a sample of MNIST image data. It is decomposed into 80 features by PCA.

        Args:
            x (np.ndarray): input data shape is (80,).
        """
        n_features = x.shape[0]

        # Group indices based on proximity
        groups = [list(range(i, min(i + self.group_size, n_features))) for i in
                  range(0, n_features, self.group_size)]
        grouped_indices = [idx for group in groups for idx in group]

        # Reorder features based on grouping
        x_reordered = x[grouped_indices]

        # Repeat the encoding and entanglement layers until all features are encoded
        for k in range(n_features // self.n_qubits):
            # Encode n features using RX gates
            for j in range(self.n_qubits):
                qml.RX(phi=x_reordered[k * self.n_qubits + j], wires=j)

            # Apply CNOT gates in a nearest-neighbor configuration
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])