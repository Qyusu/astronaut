import numpy as np
import yaml


def get_n_qubits_from_config(config_path: str) -> int:
    config = yaml.safe_load(open(config_path))
    n_qubits = config.get("device", {}).get("n_qubits", None)
    if n_qubits is None:
        raise ValueError("n_qubits is not found in the config.")
    return n_qubits


def get_molname_from_config(config_path: str) -> str:
    config = yaml.safe_load(open(config_path))
    molname = (
        config.get("hamiltonian", {}).get("params", {}).get("molname", None)
    )
    if molname is None:
        raise ValueError("molname is not found in the config.")
    return molname


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


def filter_truncated_bond_lengths(
    bond_lengths: list[float], decimals: int
) -> list[float]:
    factor = 10**decimals
    truncated = np.floor(np.array(bond_lengths) * factor) / factor

    return [bl for bl, tr in zip(bond_lengths, truncated) if np.isclose(bl, tr)]
