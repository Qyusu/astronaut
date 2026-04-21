import pytest

from astronaut.logics.vqe.molecule_configs import (
    BondLengthSettings,
    EnergyCache,
    Structures,
    TargetMolecule,
)
from astronaut.logics.vqe.utils import (
    get_bond_lengths,
    get_coordinates,
    update_config_for_molecule,
)


def test_update_config_for_molecule() -> None:
    # Setup mock molecule settings
    molecule_settings = TargetMolecule(
        name="H2",
        symbols=["H", "H"],
        n_qubits=4,
        structures=Structures(
            charge=0,
            multi=1,
            active_electrons=2,
            active_orbitals=2,
        ),
        bond_length_settings=BondLengthSettings(
            min_bond_length=0.2,
            max_bond_length=2.5,
            sample_nums=10,
            equilibrium_bond_length=0.741,
        ),
        exact_energy="fci_energy",
        energy_cache={
            0.741: EnergyCache(
                hf=-1.1167,
                casci=-1.1372,
                casscf=-1.1372,
                fci=-1.1372,
            )
        },
    )

    # Base config structure
    config: dict[str, dict] = {
        "hamiltonian": {"params": {}},
        "device": {},
    }

    bond_length = 0.741

    update_config_for_molecule(config, molecule_settings, bond_length)

    # Assertions
    assert config["hamiltonian"]["params"]["symbols"] == ["H", "H"]
    assert config["hamiltonian"]["params"]["coordinates"] == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.741],
    ]
    assert config["hamiltonian"]["params"]["charge"] == 0
    assert config["hamiltonian"]["params"]["multi"] == 1
    assert config["hamiltonian"]["params"]["active_electrons"] == 2
    assert config["hamiltonian"]["params"]["active_orbitals"] == 2

    assert config["device"]["n_qubits"] == 4

    # Energy cache check
    assert config["hamiltonian"]["params"]["hf_energy"] == -1.1167
    assert config["hamiltonian"]["params"]["casci_energy"] == -1.1372
    assert config["hamiltonian"]["params"]["casscf_energy"] == -1.1372
    assert config["hamiltonian"]["params"]["fci_energy"] == -1.1372


def test_update_config_for_molecule_no_energy_cache() -> None:
    # Setup mock molecule settings without cache hit
    molecule_settings = TargetMolecule(
        name="H2",
        symbols=["H", "H"],
        n_qubits=4,
        structures=Structures(
            charge=0,
            multi=1,
            active_electrons=2,
            active_orbitals=2,
        ),
        bond_length_settings=BondLengthSettings(
            min_bond_length=0.2,
            max_bond_length=2.5,
            sample_nums=10,
            equilibrium_bond_length=0.741,
        ),
        exact_energy="fci_energy",
        energy_cache={},  # Empty cache
    )

    config: dict[str, dict] = {
        "hamiltonian": {"params": {}},
        "device": {},
    }

    bond_length = 1.0  # arbitrary value not in cache

    update_config_for_molecule(config, molecule_settings, bond_length)

    # Basic assertions
    assert config["hamiltonian"]["params"]["symbols"] == ["H", "H"]
    assert config["device"]["n_qubits"] == 4

    # Ensure energy keys are NOT added
    assert "hf_energy" not in config["hamiltonian"]["params"]


def test_get_bond_lengths() -> None:
    # Without equilibrium_bond_length
    bls = get_bond_lengths(0.5, 1.5, 3)
    assert bls == [0.5, 1.0, 1.5]

    # With equilibrium_bond_length
    bls = get_bond_lengths(0.5, 1.5, 4, equilibrium_bond_length=0.7)
    assert bls == [0.5, 0.7, 1.0, 1.5]


def test_get_coordinates() -> None:
    coords = get_coordinates("H2", 0.74)
    assert coords == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]

    coords = get_coordinates("LiH", 1.6)
    assert coords == [[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]]

    coords = get_coordinates("BeH2", 1.3)
    assert coords == [[0.0, 0.0, -1.3], [0.0, 0.0, 0.0], [0.0, 0.0, 1.3]]

    coords = get_coordinates("H2O", 0.96)
    assert len(coords) == 3
    assert coords[0] == [0.0, 0.0, 0.0]
    # Just check if second and third atoms are symmetric
    assert coords[1][1] == -coords[2][1]
    assert coords[1][2] == coords[2][2]

    coords = get_coordinates("N2", 1.1)
    assert coords == [[0.0, 0.0, -0.55], [0.0, 0.0, 0.55]]

    coords = get_coordinates("H4", 1.0)
    assert coords == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]

    with pytest.raises(ValueError, match="Unsupported molecule"):
        get_coordinates("Unknown", 1.0)
