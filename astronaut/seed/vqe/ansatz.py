import pennylane as qml
from qxmt.ansatze import BaseVQEAnsatz
from qxmt.devices import BaseDevice
from qxmt.hamiltonians.pennylane.molecular import MolecularHamiltonian

# above are the default imports. DO NOT REMOVE THEM.
# new imports can be added below this line if needed.


class BaseAnsatz(BaseVQEAnsatz):
    """Base ansatz class.

    Args:
        BaseVQEAnsatz (_type_): base ansatz class

    Example:
    """

    def __init__(
        self, device: BaseDevice, hamiltonian: MolecularHamiltonian
    ) -> None:
        super().__init__(device, hamiltonian)
        self.hamiltonian = hamiltonian
        self.n_qubits = self.hamiltonian.get_n_qubits()
        self.n_params = 1  # number of parameters to be optimized. This value calculated in init method.

    def circuit(self, params: qml.numpy.ndarray) -> None:
        """Define your ansatz here.

        Args:
            params (qml.numpy.ndarray): parameters to be optimized. The shape is (n_params,).
                Note:
                    The 'params' argument is directly updated by the optimization library.
                    Do not reassign or modify 'params' itself.
                    When using the parameters, access them by their index (e.g., params[0], params[1], ...).
        """
        # define your ansatz below this line
        pass
