from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from astronaut.db.pennylane.setup_db import PennylaneCodeDB


@pytest.fixture
def mock_pennylane_db(mocker: MockerFixture) -> PennylaneCodeDB:
    mocker.patch("astronaut.db.pennylane.setup_db.EmbeddingClient")
    mocker.patch("astronaut.db.pennylane.setup_db.PineconeClient")
    mocker.patch("astronaut.configs.settings.EMBEDDING_MODEL_VERSION", "v1")
    mocker.patch("astronaut.configs.settings.PINECONE_API_KEY", "key")

    return PennylaneCodeDB(
        index_name="test-index", chunk_size=100, init_db=False
    )


def test_setup(
    mocker: MockerFixture, mock_pennylane_db: PennylaneCodeDB
) -> None:
    mock_pennylane_db.db.create_index = mocker.Mock()  # type: ignore
    mock_pennylane_db.setup()
    mock_pennylane_db.db.create_index.assert_called_once()


def test_upsert_full_code(
    mocker: MockerFixture, mock_pennylane_db: PennylaneCodeDB, tmp_path: Path
) -> None:
    # Create temp file
    file_path = tmp_path / "test_file.py"
    with open(file_path, "w") as f:
        f.write("print('hello')")

    mock_pennylane_db.db.upsert = mocker.Mock()  # type: ignore
    mocker.patch("builtins.open", mocker.mock_open(read_data="print('hello')"))

    mock_pennylane_db.upsert_full_code("/path/to/pennylane/test_file.py")

    mock_pennylane_db.db.upsert.assert_called_once()
    call_args = mock_pennylane_db.db.upsert.call_args[1]
    assert call_args["document_id"].endswith("test_file.py")
    assert call_args["text"] == "print('hello')"


def test_extract_classes_with_docstrings(
    mock_pennylane_db: PennylaneCodeDB, tmp_path: Path
) -> None:
    code = """
class MyClass:
    '''My Docstring'''
    pass

class NoDoc:
    pass
    """
    file_path = tmp_path / "test_classes.py"
    with open(file_path, "w") as f:
        f.write(code)

    docs = mock_pennylane_db.extract_classes_with_docstrings(str(file_path))

    assert "MyClass" in docs
    assert docs["MyClass"] == "My Docstring"
    assert "NoDoc" not in docs


def test_upsert_class_doc(
    mocker: MockerFixture, mock_pennylane_db: PennylaneCodeDB
) -> None:
    mocker.patch.object(
        mock_pennylane_db,
        "extract_classes_with_docstrings",
        return_value={"MyClass": "My Docstring"},
    )
    mock_pennylane_db.db.upsert = mocker.Mock()  # type: ignore

    mock_pennylane_db.upsert_class_doc("/path/to/pennylane/file.py")

    mock_pennylane_db.db.upsert.assert_called_once()
    call_args = mock_pennylane_db.db.upsert.call_args[1]
    assert call_args["document_id"] == "MyClass"
    assert call_args["text"] == "My Docstring"
