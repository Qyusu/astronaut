def load_code(file_path: str) -> str:
    with open(file_path, "r") as file:
        code = file.read()
    return code


def save_generated_code(save_path: str, code: str) -> None:
    with open(save_path, "w") as file:
        file.write(code)
