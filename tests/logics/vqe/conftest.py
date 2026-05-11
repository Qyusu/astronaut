import pytest


@pytest.fixture
def base_config():
    """
    Base configuration for VQE tests.
    This fixture provides the standard astronaut.yaml configuration structure.
    """
    return {
        "description": "This is a template configuration file for the QXMT framework.",  # noqa: E501
        "global_settings": {"random_seed": 42, "model_type": "vqe"},
        "hamiltonian": {
            "module_name": "qxmt.hamiltonians.pennylane",
            "implement_name": "MolecularHamiltonian",
            "params": {
                "molname": "H6",
                "basis_name": "STO-3G",
                "bondlength": 0.8,
                "active_electrons": 6,
                "active_orbitals": 6,
            },
        },
        "device": {
            "platform": "pennylane",
            "device_name": "lightning.qubit",
            "n_qubits": 12,
            "shots": None,
            "random_seed": 42,
        },
        "ansatz": {
            "module_name": "astronaut.seed.vqe.ansatz",
            "implement_name": "BaseAnsatz",
            "params": None,
        },
        "model": {
            "name": "basic",
            "diff_method": "best",
            "optimizer_settings": {"name": "scipy.SLSQP", "params": None},
            "params": {
                "max_steps": 2000,
                "init_params": {"type": "zeros"},
                "verbose": False,
            },
        },
        "evaluation": {
            "default_metrics": ["final_cost", "hf_energy", "fci_energy"],
            "custom_metrics": [],
        },
    }
