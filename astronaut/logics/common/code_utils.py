from pathlib import Path


def load_code(file_path: str) -> str:
    with open(file_path, "r") as file:
        code = file.read()
    return code


def save_generated_code(save_path: str, code: str) -> None:
    with open(save_path, "w") as file:
        file.write(code)


def insert_molecule_name_to_path(file_path: str, molecule_name: str) -> str:
    """Insert molecule name into file path before the extension.

    Args:
        file_path: Original file path (e.g., "../hoge/hoge/astronaut.yaml")
        molecule_name: Name to insert (e.g., "test")

    Returns:
        Modified file path with molecule_name inserted (e.g., "../hoge/hoge/astronaut-test.yaml")

    Example:
        >>> insert_molecule_name_to_path("../hoge/hoge/astronaut.yaml", "test")
        "../hoge/hoge/astronaut-test.yaml"
    """
    if molecule_name == "":
        return file_path

    path = Path(file_path)
    updated_file_name = f"{path.stem}-{molecule_name}{path.suffix}"
    return str(path.parent / updated_file_name)
