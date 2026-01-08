"""Embedding service for JAVIS memory and RAG systems."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid loading heavy dependencies at startup
_sentence_transformer = None


class EmbeddingService:
    """Sentence-transformers based embedding service (singleton)."""

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding service.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self._model = None
        self._dimension: Optional[int] = None

    @classmethod
    def get_instance(
        cls, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> "EmbeddingService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded. Dimension: {self._dimension}")
            except ImportError:
                logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        self._load_model()

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        result = self.embed([text])
        return result[0] if result else []

    def similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


def get_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> EmbeddingService:
    """Get the global embedding service instance."""
    return EmbeddingService.get_instance(model_name)
