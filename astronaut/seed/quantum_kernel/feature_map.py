import numpy as np
from qxmt.constants import PENNYLANE_PLATFORM
from qxmt.feature_maps import BaseFeatureMap

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class SeedFeatureMap(BaseFeatureMap):
    """Seed feature map class.

    Args:
        BaseFeatureMap (_type_): base feature map class

    Example:
    """

    def __init__(self, n_qubits: int) -> None:
        """Initialize the Seed feature map class.

        Args:
            n_qubits (int): number of qubits
        """
        super().__init__(PENNYLANE_PLATFORM, n_qubits)
        # hyperparameters
        self.n_qubits: int = n_qubits

    def feature_map(self, x: np.ndarray) -> None:
        """Create quantum circuit of feature map.
        The input data is a single feature vector.

        Args:
            x (np.ndarray): single feature vector
        """
        # define your quantum feature map below this line
        pass
