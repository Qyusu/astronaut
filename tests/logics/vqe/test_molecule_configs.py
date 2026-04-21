import pytest
from pydantic import ValidationError

from astronaut.logics.vqe.molecule_configs import (
    BondLengthSettings,
    EnergyCache,
    HamiltonianSettings,
    Structures,
    TargetMolecule,
    get_molecule_config,
    get_molecule_config_from_qubits,
)


def test_bond_length_settings_validation() -> None:
    # Valid data
    settings = BondLengthSettings(
        min_bond_length=0.1,
        max_bond_length=1.5,
        sample_nums=10,
        equilibrium_bond_length=0.74,
    )
    assert settings.min_bond_length == 0.1

    # Invalid data (negative values)
    with pytest.raises(ValidationError):
        BondLengthSettings(
            min_bond_length=-0.1,
            max_bond_length=1.5,
            sample_nums=10,
            equilibrium_bond_length=0.74,
        )

    # Invalid data (sample_nums <= 0)
    with pytest.raises(ValidationError):
        BondLengthSettings(
            min_bond_length=0.1,
            max_bond_length=1.5,
            sample_nums=0,
            equilibrium_bond_length=0.74,
        )


def test_structures_validation() -> None:
    # Valid data
    struct = Structures(
        charge=0,
        multi=1,
        active_electrons=2,
        active_orbitals=2,
    )
    assert struct.charge == 0

    # Invalid data (negative active_electrons)
    with pytest.raises(ValidationError):
        Structures(
            charge=0,
            multi=1,
            active_electrons=-1,
            active_orbitals=2,
        )


def test_hamiltonian_settings_defaults() -> None:
    settings = HamiltonianSettings()
    assert settings.basis_name == "STO-3G"
    assert settings.method == "pyscf"
    assert settings.unit == "angstrom"


def test_target_molecule_and_utils() -> None:
    h2_config = TargetMolecule(
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
    )

    lih_config = TargetMolecule(
        name="LiH",
        symbols=["Li", "H"],
        n_qubits=12,
        structures=Structures(
            charge=0,
            multi=1,
            active_electrons=2,
            active_orbitals=6,
        ),
        bond_length_settings=BondLengthSettings(
            min_bond_length=0.5,
            max_bond_length=3.0,
            sample_nums=10,
            equilibrium_bond_length=1.595,
        ),
        exact_energy="fci_energy",
    )

    molecule_configs = {
        "H2": h2_config,
        "LiH": lih_config,
    }

    # Test get_molecule_config
    assert get_molecule_config("H2", molecule_configs) == h2_config
    assert get_molecule_config("LiH", molecule_configs) == lih_config
    with pytest.raises(ValueError, match="Unsupported molecule name: O2"):
        get_molecule_config("O2", molecule_configs)

    # Test get_molecule_config_from_qubits
    assert get_molecule_config_from_qubits(4, molecule_configs) == h2_config
    assert get_molecule_config_from_qubits(12, molecule_configs) == lih_config
    assert get_molecule_config_from_qubits(8, molecule_configs) is None


def test_energy_cache() -> None:
    cache = EnergyCache(hf=-1.11, fci=-1.13)
    assert cache.hf == -1.11
    assert cache.fci == -1.13
    assert cache.casci is None
