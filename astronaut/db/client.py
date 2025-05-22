import re
import time

import tiktoken
from loguru import logger
from pinecone import Pinecone, QueryResponse, ServerlessSpec

from astronaut.llm import EmbeddingClient

TIKTOKEN_MODEL = "cl100k_base"


class PineconeClient:
    """A client for interacting with Pinecone vector database.

    This class provides functionality to create, manage, and query vector indexes in Pinecone.
    It supports document chunking, embedding generation, and vector similarity search.

    Args:
        api_key (str): Pinecone API key for authentication
        index_name (str): Name of the Pinecone index to use
        embed_client (EmbeddingClient): Client for generating text embeddings

    Attributes:
        pc (Pinecone): Pinecone client instance
        embed_client (EmbeddingClient): Client for generating embeddings
        total_cost (float): Total cost incurred for embedding operations
        index_name (str): Name of the current index
        index (Pinecone.Index): Active Pinecone index instance

    Methods:
        create_index: Creates a new Pinecone index with specified dimension and metric
        _chunk_by_size: Splits text into chunks based on token size
        _chunk_by_sentence_and_size: Splits text into chunks based on sentences
        upsert: Upserts a document into the index by splitting it into chunks
        query: Queries the index for similar vectors based on the input text
        delete_index: Deletes the specified index from Pinecone
        check_connection: Checks the connection to the Pinecone index
    """

    def __init__(self, api_key: str, index_name: str, embed_client: EmbeddingClient) -> None:
        self.pc = Pinecone(api_key=api_key)
        self.embed_client = embed_client
        self.total_cost = 0.0
        self.index_name = index_name
        if index_name in self.pc.list_indexes().names():
            self.index = self.pc.Index(index_name)
        else:
            self.index = None

    def create_index(self, dimension: int, metric: str = "cosine") -> None:
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Index {self.index_name} is being created. This may take a few minutes.")
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
            logger.info(f"Index {self.index_name} has been created.")
        else:
            logger.info(f"Index {self.index_name} already exists.")

        self.index = self.pc.Index(self.index_name)

    def _chunk_by_size(self, text: str, chunk_size: int) -> list[str]:
        tokenizer = tiktoken.get_encoding(TIKTOKEN_MODEL)
        tokens = tokenizer.encode(text)
        chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        return [tokenizer.decode(chunk) for chunk in chunks]

    def _chunk_by_sentence_and_size(self, text: str, chunk_size: int) -> list[str]:
        tokenizer = tiktoken.get_encoding(TIKTOKEN_MODEL)
        sentences = re.split(r"(?<=[.!?])\s+", text)  # splite by sentence
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            tokens = tokenizer.encode(current_chunk + " " + sentence)
            if len(tokens) <= chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence  # start a new chunk

        # add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def upsert(
        self,
        document_id: str,
        text: str,
        chunk_size: int,
        metadata: dict = {},
        chunk_method: str = "size",
        allow_update: bool = False,
    ) -> None:
        if self.index is None:
            raise ValueError("Index is not created.")

        existing_data = self.index.fetch(ids=[document_id])
        if (not allow_update) and (existing_data.get("vectors")):
            logger.error(f"Document {document_id} already exists in the index. Not upserted.")
            return None

        if chunk_method == "size":
            chunks = self._chunk_by_size(text, chunk_size)
        elif chunk_method == "sentence":
            chunks = self._chunk_by_sentence_and_size(text, chunk_size)
        else:
            raise ValueError(f"Invalid chunk method: {chunk_method}")

        chunks_vector, cost = self.embed_client.embeddings(chunks)
        self.total_cost += cost
        chunk_list = []
        for c_id, (c_text, c_vector) in enumerate(zip(chunks, chunks_vector)):
            metadata = metadata | ({"chunk_text": "".join(c_text), "chunk_id": c_id})
            chunk_list.append((f"{document_id}_{c_id}", c_vector, metadata))
            if len(chunk_list) == 100:  # upsert every 100 chunks
                self.index.upsert(chunk_list)
                chunk_list = []
        self.index.upsert(chunk_list)
        logger.info(f"Document {document_id} has been upserted. Total {len(chunks)} chunks.")

    def query(self, text: str, top_k: int = 5, metadata_filter: dict = {}) -> QueryResponse:
        if self.index is None:
            raise ValueError("Index is not created.")

        vector, cost = self.embed_client.embeddings([text])
        self.total_cost += cost
        response = self.index.query(vector=vector[0], top_k=top_k, include_metadata=True, filter=metadata_filter)
        if isinstance(response, QueryResponse):
            return response
        else:
            raise TypeError("Expected QueryResponse, got {type(response).__name__}")

    def delete_index(self, index_name: str) -> None:
        exist_index_names = [index["name"] for index in self.pc.list_indexes()]
        if index_name in exist_index_names:
            self.pc.delete_index(index_name)
            self.index = None
            logger.info(f'Index "{index_name}" has been deleted.')
        else:
            logger.info(f'Index "{index_name}" does not exist.')

    def check_connection(self) -> bool:
        try:
            if self.index is None:
                logger.error("Index is not created.")
                return False

            index_stats = str(self.index.describe_index_stats()).replace("\n", " ")
            logger.info(f'Index "{self.index_name}" is connected.')
            logger.info(f"Index stats: {index_stats}")
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False
