"""
Embeddings Module
Uses local HuggingFace embedding models (no API limits).
"""

import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class LocalEmbeddings:
    """
    Local embedding model using HuggingFace Sentence Transformers.
    No API limits, runs completely offline on your CPU/GPU.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            # model_kwargs={'device': 'cpu'},  # Can omit to auto-detect GPU/CPU
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Initialized Local Embeddings: {self.model_name}")

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Return the underlying LangChain embeddings object."""
        return self._embeddings

    def get_query_embeddings_instance(self) -> HuggingFaceEmbeddings:
        """Return an embeddings instance configured for query-time use."""
        return self._embeddings

    def get_embedding_dimension(self) -> int:
        """Return the embedding dimension for bge-small-en (384)."""
        return 384
