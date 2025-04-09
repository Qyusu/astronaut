import ast
import os
from typing import Literal

import click

from astronaut.configs import settings
from astronaut.db.client import PineconeClient
from astronaut.llm.embedding import EmbeddingClient


class PennylaneCodeDB:
    """A class for managing and storing Pennylane API reference in a vector database.

    This class provides functionality to extract and store Pennylane API documentation
    in a vector database (Pinecone) with text embeddings. It supports two types of
    documentation storage:
    1. Source code: Stores the complete Pennylane API code
    2. Class docstring: Stores class names and their docstrings

    Currently, the Pennylane API reference is extracted from the Pennylane codebase
    version 0.39.0, which matches the QXMT used version.

    Args:
        index_name (str): Name of the Pinecone index to use
        chunk_size (int): Size of text chunks for embedding
        init_db (bool): Whether to initialize/clear the database before operations

    Attributes:
        index_name (str): Name of the current index
        chunk_size (int): Size of text chunks for embedding
        init_db (bool): Database initialization flag
        client (EmbeddingClient): Client for generating text embeddings
        db (PineconeClient): Client for vector database operations

    Methods:
        setup: Initializes the vector database
        upsert_full_code: Upserts complete Pennylane API code to the database
        extract_classes_with_docstrings: Extracts class names and docstrings from Python files
        upsert_class_doc: Upserts class documentation to the database
        process_code_in_directory: Processes all Python files in a directory

    Reference:
        - Pennylane API: https://docs.pennylane.ai/en/stable/code/qml.html
    """

    def __init__(self, index_name: str, chunk_size: int, init_db: bool) -> None:
        self.index_name = index_name
        self.chunk_size = chunk_size
        self.init_db = init_db

        self.client = EmbeddingClient(
            platform=settings.EMBEDDING_PLATFORM,
            api_key=settings.OPENAI_API_KEY if settings.EMBEDDING_PLATFORM == "openai" else "",
            embeddings_model_version=settings.EMBEDDING_MODEL_VERSION,
        )
        self.db = PineconeClient(
            api_key=settings.PINECONE_API_KEY, index_name=self.index_name, embed_client=self.client
        )

    def setup(self) -> None:
        if self.init_db:
            self.db.delete_index(self.index_name)

        self.db.create_index(
            dimension=settings.EMBEDDING_DIM,
            metric="cosine",
        )

    def upsert_full_code(self, full_file_path: str) -> None:
        with open(full_file_path, "r", encoding="utf-8") as f:
            code = f.read()

        file_path = "pennylane" + full_file_path.split("pennylane")[-1]
        metadata = {"file_path": file_path}
        self.db.upsert(
            document_id=file_path,
            text=code,
            chunk_size=self.chunk_size,
            metadata=metadata,
            chunk_method="size",
            allow_update=False,
        )

    def extract_classes_with_docstrings(self, full_file_path: str) -> dict:
        with open(full_file_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read())

        class_docs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    class_docs[node.name] = docstring

        return class_docs

    def upsert_class_doc(self, full_file_path: str) -> None:
        class_docs = self.extract_classes_with_docstrings(full_file_path)
        file_path = "pennylane" + full_file_path.split("pennylane")[-1]

        for class_name, class_doc in class_docs.items():
            metadata = {"file_path": file_path, "class_name": class_name, "call_name": f"qml.{class_name}"}
            self.db.upsert(
                document_id=class_name,
                text=class_doc,
                chunk_size=self.chunk_size,
                metadata=metadata,
                chunk_method="size",
                allow_update=False,
            )

    def process_code_in_directory(self, directory: str, docs_type: Literal["source_code", "class_doc"]) -> None:
        for root, _, files in os.walk(os.path.expanduser(directory)):
            for file_name in files:
                if file_name.endswith(".py"):
                    file_path = os.path.join(root, file_name)
                    if docs_type == "source_code":
                        self.upsert_full_code(file_path)
                    elif docs_type == "class_doc":
                        self.upsert_class_doc(file_path)


@click.command()
@click.option("--chunk_size", type=int, default=512, required=False, help="The size of the chunk.")
@click.option("--init_db", type=bool, default=False, required=False, help="Initialize the database.")
@click.option("--code_dirc", type=str, required=True, help="The directory path of root pennylane code.")
@click.option("--docs_type", type=str, required=True, help="upsert docs type: source_code or class_doc")
def setup_pennylane_db(
    chunk_size: int,
    init_db: bool,
    code_dirc: str,
    docs_type: Literal["source_code", "class_doc"],
) -> None:
    pennylane_db = PennylaneCodeDB(index_name=settings.PENNLYLANE_INDEX_NAME, chunk_size=chunk_size, init_db=init_db)
    pennylane_db.setup()
    pennylane_db.process_code_in_directory(code_dirc, docs_type)


if __name__ == "__main__":
    setup_pennylane_db()
