from typing import Any

import numpy as np

import astronaut.logics.vqe.constants as vqe_const
from astronaut.logics.vqe.molecule_configs import TargetMolecule


def get_bond_lengths(
    min_bond_length: float,
    max_bond_length: float,
    sample_nums: int,
    equilibrium_bond_length: float | None = None,
) -> list[float]:
    if equilibrium_bond_length is None:
        bond_lengths = np.linspace(
            min_bond_length, max_bond_length, sample_nums
        ).tolist()
    else:
        bond_lengths = np.linspace(
            min_bond_length, max_bond_length, sample_nums - 1
        ).tolist()
        bond_lengths.append(equilibrium_bond_length)

    return sorted(bond_lengths)


def get_coordinates(
    molecule_name: str, bond_length: float
) -> list[list[float]]:
    """
    Return 3D coordinates (in angstroms) for a given molecule and bond length.

    Parameters:
        molecule_name (str): Molecule name, e.g., "H2", "LiH", "BeH2", "H2O", "N2", "H4", ...
        bond_length (float): Bond length to use in angstrom

    Returns:
        List[List[float]]: Coordinates of atoms in Cartesian format
    """
    match molecule_name:
        case "H2":
            return [[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]]

        case "LiH":
            return [[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]]

        case "BeH2":
            return [
                [0.0, 0.0, -bond_length],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, bond_length],
            ]

        case "H2O":
            angle_deg = 104.5
            angle_rad = np.radians(angle_deg / 2)
            x = float(bond_length * np.sin(angle_rad))
            z = float(bond_length * np.cos(angle_rad))
            return [[0.0, 0.0, 0.0], [0.0, x, z], [0.0, -x, z]]  # O, H1, H2

        case "N2":
            return [[0.0, 0.0, -bond_length / 2], [0.0, 0.0, bond_length / 2]]

        case "H4" | "H6" | "H8" | "H10":
            n = int(molecule_name[1:])
            # Linear chain: H-H-H-... with uniform spacing
            return [[i * bond_length, 0.0, 0.0] for i in range(n)]

        case _:
            raise ValueError(f"Unsupported molecule: {molecule_name}")


def update_config_for_molecule(
    config: dict[str, Any],
    molecule_settings: TargetMolecule,
    bond_length: float,
) -> None:
    """Update config with molecule settings."""
    params = {
        vqe_const.QXMT_CONFIG_KEY_SYMBOLS: molecule_settings.symbols,
        vqe_const.QXMT_CONFIG_KEY_COORDINATES: get_coordinates(
            molecule_settings.name, bond_length
        ),
        **molecule_settings.structures.model_dump(),
        **molecule_settings.hamiltonian_settings.model_dump(),
    }

    energy_cache = molecule_settings.energy_cache
    if energy_cache and bond_length in energy_cache:
        energies = energy_cache[bond_length].model_dump(exclude_none=True)
        params.update({f"{k}_energy": v for k, v in energies.items()})

    config[vqe_const.QXMT_CONFIG_KEY_HAMILTONIAN][
        vqe_const.QXMT_CONFIG_KEY_PARAMS
    ] = params
    config[vqe_const.QXMT_CONFIG_KEY_DEVICE][
        vqe_const.QXMT_CONFIG_KEY_N_QUBITS
    ] = molecule_settings.n_qubits
