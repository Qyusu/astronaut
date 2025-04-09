import time

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from loguru import logger
from openai import OpenAI
from openai.types.create_embedding_response import CreateEmbeddingResponse

from astronaut.llm.cost import EmbeddingModelCostTable


class EmbeddingClient:
    """Client for generating text embeddings using OpenAI's embedding models.

    This class provides functionality for converting text into vector embeddings
    using OpenAI's embedding models. It handles API requests, response parsing,
    and cost tracking specific to the embedding generation process.

    The class supports features like:
    - Batch processing of text inputs
    - Automatic retries with exponential backoff
    - Cost calculation based on token usage
    - LangSmith tracing for monitoring

    Args:
        platform (str): LLM platform to use (currently only "openai" is supported)
        api_key (str): OpenAI API key for authentication
        embeddings_model_version (str): Model version to use for generating embeddings

    Attributes:
        platform (str): Selected LLM platform
        api_key (str): API key for authentication
        embeddings_model_version (str): Model version for embeddings
        client (OpenAI): OpenAI API client instance

    Methods:
        embeddings: Main method for generating embeddings from text
        _embeddings: Helper method for API requests with tracing
        _calculate_cost: Calculates the cost of embedding generation
        _initialize_client: Creates the appropriate API client

    Note:
        Currently, this client only supports OpenAI's embedding models.
        Support for other platforms may be added in the future.

    Reference:
        GitHub: https://github.com/openai/openai-python
    """

    def __init__(
        self,
        platform: str,
        api_key: str,
        embeddings_model_version: str,
    ) -> None:
        self.platform = platform.lower()
        self.api_key = api_key
        self.embeddings_model_version = embeddings_model_version
        self.client = self._initialize_client()

    def _initialize_client(self) -> OpenAI:
        try:
            if self.platform == "openai":
                return wrap_openai(OpenAI(api_key=self.api_key))
            else:
                raise ValueError(f"Embedding model only supports OpenAI. {self.platform} is not supported.")
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {e}")

    @traceable(tags=["llm"], run_type="llm")
    def _embeddings(self, model_version: str, text_list: list[str]) -> CreateEmbeddingResponse:
        response = self.client.embeddings.create(
            model=model_version,
            input=text_list,
        )
        return response

    def _calculate_cost(self, response: CreateEmbeddingResponse) -> float:
        model_version = self.embeddings_model_version
        cost_per_1m_tokens = EmbeddingModelCostTable().get_cost(model_version)
        if cost_per_1m_tokens is None:
            logger.info(f'Model version "{model_version}" is not found in the cost table.')
            return 0.0

        usage = response.usage
        if usage is None:
            logger.info("Usage information is not found in the response.")
            return 0.0

        cost = usage.total_tokens * cost_per_1m_tokens.input / 10**6

        return cost

    def embeddings(self, text_list: list[str], max_retries: int = 3) -> tuple[list[list[float]], float]:
        """Generate embeddings for a list of text inputs.

        This method converts a list of text strings into their corresponding vector embeddings
        using the configured OpenAI embedding model. It handles the entire process including
        API requests, response parsing, and error handling with automatic retries.

        The method implements exponential backoff for retries, with the wait time doubling
        after each failed attempt. This helps handle temporary API issues and rate limits.

        Args:
            text_list (list[str]): List of text strings to convert into embeddings.
                Each string will be converted into a vector representation.
            max_retries (int, optional): Maximum number of retry attempts for failed API calls.
                Defaults to 3. Each retry will wait longer than the previous one.

        Returns:
            tuple[list[list[float]], float]: A tuple containing:
                - List of embedding vectors, where each vector corresponds to the input text
                - Total cost of the API request in USD

        Raises:
            ValueError: If all retry attempts fail to get a valid response from the API.
                The error message will include details about the final failure.

        Note:
            The method uses LangSmith tracing for monitoring and debugging purposes.
            The cost calculation is based on the token usage reported by the API.
        """
        attempts = 0
        while True:
            try:
                response = self._embeddings(model_version=self.embeddings_model_version, text_list=text_list)
                cost = self._calculate_cost(response)
                return [record.embedding for record in response.data], cost
            except Exception as e:
                logger.info(f"Raise Exception: {e}")
                attempts += 1
                if attempts >= max_retries:
                    raise ValueError(f"Failed to get embeddings from OpenAI after {max_retries} attempts: {e}")

                wait_time = 60 * 2**attempts
                logger.info(f"Retry after {wait_time} seconds...")
                time.sleep(wait_time)
