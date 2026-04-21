from pathlib import Path

import pdfplumber
import tiktoken
from loguru import logger


class Paper:
    """
    Base class for handling academic papers.

    This class provides common functionality for processing academic papers,
    including PDF loading, text processing, and token management.

    Args:
        local_paper_dir (str): Path to the local directory containing paper PDFs.

    Attributes:
        local_paper_dir (str): Directory containing local PDFs.

    Methods:
        load_full_text_from_local(paper_id):
            Loads the full text of a paper from a local PDF file.
        cut_string_if_over_token_limit(text, max_tokens, model_name):
            Cuts a string if it exceeds the specified token limit.
    """

    def __init__(self, local_paper_dir: str) -> None:
        self.local_paper_dir = local_paper_dir

    def load_full_text_from_local(self, paper_id: str) -> str:
        """
        Load the full text of a paper from a local PDF file.

        Args:
            paper_id (str): The ID of the paper to load.

        Returns:
            str: The full text content of the paper.

        Raises:
            ValueError: If local paper directory is not provided.
        """
        local_pdf_path = Path(f"{self.local_paper_dir}/{paper_id}.pdf")
        if not local_pdf_path.exists():
            logger.warning(f"PDF file not found: {local_pdf_path}")
            return ""

        text_content = []
        try:
            with pdfplumber.open(local_pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_content.append(text)
            full_text = "\n".join(text_content)
            logger.debug(f"Successfully loaded text from {paper_id}.pdf")
            return full_text
        except Exception as e:
            logger.error(f"Error loading PDF {paper_id}: {e}")
            return ""

    def cut_string_if_over_token_limit(
        self,
        text: str,
        max_tokens: int,
        model_name: str = "gpt-4o",
    ) -> str:
        """
        Cut a string if it exceeds the specified token limit.

        Args:
            text (str): The text to potentially cut.
            max_tokens (int): The maximum number of tokens allowed.
            model_name (str): The model name to use for token encoding.

        Returns:
            str: The potentially truncated text.
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            tokens = encoding.encode(text)
            if len(tokens) > max_tokens:
                logger.info(
                    f"Text truncated from {len(tokens)} to {max_tokens} tokens"
                )
                tokens = tokens[:max_tokens]
            return encoding.decode(tokens)
        except Exception as e:
            logger.error(f"Error in token processing: {e}")
            return text

    def get_paper_path(self, paper_id: str) -> Path:
        """
        Get the path to a paper PDF file.

        Args:
            paper_id (str): The ID of the paper.

        Returns:
            Path: The path to the paper PDF file.

        Raises:
            ValueError: If local paper directory is not provided.
        """
        return Path(f"{self.local_paper_dir}/{paper_id}.pdf")

    def paper_exists(self, paper_id: str) -> bool:
        """
        Check if a paper PDF file exists locally.

        Args:
            paper_id (str): The ID of the paper to check.

        Returns:
            bool: True if the paper exists, False otherwise.
        """
        try:
            return self.get_paper_path(paper_id).exists()
        except ValueError:
            return False
