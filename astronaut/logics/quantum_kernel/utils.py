import yaml


def get_n_qubits_from_config(config_path: str) -> int:
    config = yaml.safe_load(open(config_path))
    n_qubits = config.get("device", {}).get("n_qubits", None)
    if n_qubits is None:
        raise ValueError("n_qubits is not found in the config.")
    return n_qubits
