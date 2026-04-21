from pathlib import Path

import click
import pennylane as qml
from loguru import logger

from astronaut.db.pennylane.format_api_docs import format_docs_by_llm
from astronaut.db.pennylane.parse_api import parse_all_modules

MODULES = [(qml, "qml"), (qml.qchem, "qml.qchem")]


@click.command()
@click.option(
    "--save_raw_json_path",
    type=click.Path(exists=False),
    required=True,
    help="Path to save the raw JSON file",
)
@click.option(
    "--save_formatted_json_path",
    type=click.Path(exists=False),
    required=True,
    help="Path to save the formatted JSON file",
)
def main(save_raw_json_path: Path, save_formatted_json_path: Path) -> None:
    logger.info(f"Pennylane version: {qml.__version__}")
    parse_all_modules(MODULES, save_raw_json_path)

    format_docs_by_llm(save_raw_json_path, save_formatted_json_path)


if __name__ == "__main__":
    main()
