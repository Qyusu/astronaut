import os
import tempfile

import pytest
import yaml

from astronaut.logics.vqe.utils import (
    filter_truncated_bond_lengths,
    get_molname_from_config,
    get_n_qubits_from_config,
)


def test_get_n_qubits_from_config(base_config):
    # Test successful case
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(base_config, f)

        n_qubits = get_n_qubits_from_config(config_path)
        assert n_qubits == 12


def test_get_n_qubits_from_config_missing_n_qubits(base_config):
    # Test when n_qubits is missing
    config_without_n_qubits = base_config.copy()
    del config_without_n_qubits["device"]["n_qubits"]

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_without_n_qubits, f)

        with pytest.raises(ValueError) as e:
            get_n_qubits_from_config(config_path)
        assert str(e.value) == "n_qubits is not found in the config."


def test_get_n_qubits_from_config_missing_device_section(base_config):
    # Test when device section is missing
    config_without_device = base_config.copy()
    del config_without_device["device"]

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_without_device, f)

        with pytest.raises(ValueError) as e:
            get_n_qubits_from_config(config_path)
        assert str(e.value) == "n_qubits is not found in the config."


def test_get_molname_from_config(base_config):
    # Test successful case
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(base_config, f)

        molname = get_molname_from_config(config_path)
        assert molname == "H6"


def test_get_molname_from_config_missing_molname(base_config):
    # Test when molname is missing
    config_without_molname = base_config.copy()
    del config_without_molname["hamiltonian"]["params"]["molname"]

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_without_molname, f)

        with pytest.raises(ValueError) as e:
            get_molname_from_config(config_path)
        assert str(e.value) == "molname is not found in the config."


def test_get_molname_from_config_missing_hamiltonian_section(base_config):
    # Test when hamiltonian section is missing
    config_without_hamiltonian = base_config.copy()
    del config_without_hamiltonian["hamiltonian"]

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_without_hamiltonian, f)

        with pytest.raises(ValueError) as e:
            get_molname_from_config(config_path)
        assert str(e.value) == "molname is not found in the config."


def test_filter_truncated_bond_lengths():
    # empty list
    bond_lengths = []
    sampled_bond_lengths = filter_truncated_bond_lengths(
        bond_lengths, decimals=1
    )
    assert sampled_bond_lengths == []

    # all elements are sampled
    bond_lengths = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30]
    sampled_bond_lengths = filter_truncated_bond_lengths(
        bond_lengths, decimals=1
    )
    assert sampled_bond_lengths == [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.10,
        1.20,
        1.30,
    ]

    # some elements are sampled
    bond_lengths = [0.50, 0.60, 0.71, 0.80, 0.95, 1.005, 1.10, 1.20, 1.30]
    sampled_bond_lengths = filter_truncated_bond_lengths(
        bond_lengths, decimals=1
    )
    assert sampled_bond_lengths == [0.50, 0.60, 0.80, 1.10, 1.20, 1.30]
