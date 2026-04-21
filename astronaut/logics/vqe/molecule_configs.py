from typing import Optional

import qxmt.hamiltonians.pennylane.molecular as qxmt_mol
from pydantic import BaseModel, Field


class BondLengthSettings(BaseModel):
    min_bond_length: float = Field(
        ..., description="Minimum bond length", ge=0.0
    )
    max_bond_length: float = Field(
        ..., description="Maximum bond length", ge=0.0
    )
    sample_nums: int = Field(..., description="Number of samples", gt=0)
    equilibrium_bond_length: float = Field(
        ..., description="Equilibrium bond length", ge=0.0
    )


class Structures(BaseModel):
    charge: int = Field(..., description="Total charge")
    multi: int = Field(..., description="Spin multiplicity")
    active_electrons: int = Field(
        ..., description="Number of active electrons", ge=0
    )
    active_orbitals: int = Field(
        ..., description="Number of active orbitals", ge=0
    )


class EnergyCache(BaseModel):
    hf: Optional[float] = Field(None, description="Hartree-Fock energy")
    casci: Optional[float] = Field(None, description="CASCI energy")
    casscf: Optional[float] = Field(None, description="CASSCF energy")
    fci: Optional[float] = Field(None, description="FCI energy")


class HamiltonianSettings(BaseModel):
    """Settings for Hamiltonian generation.

    Attributes:
        basis_name: Basis set name.
        method: Calculation method or backend (e.g., "pyscf", "openfermion").
        unit: Length unit for geometry (e.g., "angstrom", "bohr").
    """

    basis_name: qxmt_mol.SUPPORTED_BASIS_NAMES = Field(
        "STO-3G", description="Basis set name (e.g., 'STO-3G')"
    )
    method: qxmt_mol.SUPPORTED_METHODS = Field(
        "pyscf", description="Calculation method (e.g., 'pyscf')"
    )
    unit: qxmt_mol.SUPPORTED_UNITS = Field(
        "angstrom", description="Length unit (e.g., 'angstrom')"
    )


class TargetMolecule(BaseModel):
    name: str = Field(..., description="Molecule name")
    symbols: list[str] = Field(..., description="List of atomic symbols")
    n_qubits: int = Field(..., description="Number of qubits", gt=0)
    structures: Structures = Field(
        ..., description="Electronic structure settings"
    )
    bond_length_settings: BondLengthSettings = Field(
        ..., description="Bond length sampling settings"
    )
    exact_energy: str = Field(..., description="Key for exact energy reference")
    hamiltonian_settings: HamiltonianSettings = Field(
        default_factory=HamiltonianSettings,
        description="Hamiltonian generation settings",
    )
    reference_energy_methods: list[
        qxmt_mol.SUPPORTED_REFERENCE_ENERGY_METHOD
    ] = Field(
        default_factory=list[qxmt_mol.SUPPORTED_REFERENCE_ENERGY_METHOD],
        description="list of reference energy methods (e.g., 'casci', 'casscf', 'fci')",
    )
    energy_cache: dict[float, EnergyCache] | None = Field(
        None, description="Precomputed energy cache keyed by bond length"
    )


def get_molecule_config(
    mol_name: str, molecule_configs: dict[str, TargetMolecule]
) -> TargetMolecule:
    """Get molecule config by molecule name.

    Args:
        mol_name (str): molecule name
        molecule_configs (dict[str, TargetMolecule]): molecule configs

    Raises:
        ValueError: if molecule name is not supported

    Returns:
        TargetMolecule: molecule config
    """
    if mol_name not in molecule_configs:
        raise ValueError(
            f"Unsupported molecule name: {mol_name}",
            f"Available molecules: {list(molecule_configs.keys())}",
        )
    return molecule_configs[mol_name]


def get_molecule_config_from_qubits(
    qubits: int, molecule_configs: dict[str, TargetMolecule]
) -> TargetMolecule | None:
    """Get molecule config by qubit number.

    Args:
        qubits (int): qubit number
        molecule_configs (dict[str, TargetMolecule]): molecule configs

    Returns:
        TargetMolecule | None: molecule config
    """
    for mol_name, mol_config in molecule_configs.items():
        if mol_config.n_qubits == qubits:
            return mol_config
    return None
