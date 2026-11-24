"""Semantic similarity engine using sentence-transformers.

This is the core engine that powers mentions(), not_mentions(), intent(),
and drift detection. Uses offline models -- no LLM judge, no API cost.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from llm_behave.engines.text import split_sentences

logger = logging.getLogger(__name__)

# Multi-model engine cache: model_name -> SemanticEngine
_engines: dict[str, SemanticEngine] = {}
# Guards engine creation and model loading so threads never load a model twice
_LOCK = threading.RLock()

# Default model -- small (80MB), fast, works offline
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticEngine:
    """Embedding-based semantic similarity engine.

    Loads a sentence-transformer model lazily on first use.
    All subsequent calls reuse the loaded model.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_name

    def _load_model(self) -> None:
        """Load the sentence-transformer model lazily (thread-safe)."""
        if self._model is not None:
            return
        with _LOCK:
            self._load_model_locked()

    def _load_model_locked(self) -> None:
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading semantic model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Semantic model loaded successfully.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic assertions. "
                "Install it with: pip install llm-behave[semantic]"
            )

    def encode(self, text: str) -> np.ndarray:
        """Encode text into an embedding vector."""
        self._load_model()
        embedding: np.ndarray = self._model.encode(text, convert_to_numpy=True)
        return embedding

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode several texts in one forward pass. Returns shape (len(texts), dim)."""
        self._load_model()
        embeddings: np.ndarray = self._model.encode(texts, convert_to_numpy=True)
        return embeddings

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Returns a float between -1 and 1 (typically 0 to 1 for natural text).
        """
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)

        # Cosine similarity
        dot = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot / (norm_a * norm_b))

    def top_sentence_similarities(
        self, text: str, query: str, k: int = 3
    ) -> list[tuple[str, float]]:
        """Return the top-k (sentence, similarity) pairs for query against text.

        Splits text into sentences, scores each against the query, and returns
        the k highest-scoring pairs sorted descending. Used both for the max
        score and for explaining *why* an assertion failed (which sentences
        were closest, and by how much).
        """
        sentences = split_sentences(text)
        if not sentences:
            return []

        # One batched forward pass for the query and every sentence
        embeddings = self.encode_batch([query] + sentences)
        query_emb, sentence_embs = embeddings[0], embeddings[1:]
        norms = np.linalg.norm(sentence_embs, axis=1) * np.linalg.norm(query_emb)
        dots = sentence_embs @ query_emb
        scores = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)

        scored = [(sentence, float(score)) for sentence, score in zip(sentences, scores)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def max_sentence_similarity(self, text: str, query: str) -> float:
        """Compute max cosine similarity between query and any sentence in text.

        Splits text into sentences and returns the highest similarity found.
        More accurate than whole-text comparison for paragraph inputs.
        """
        top = self.top_sentence_similarities(text, query, k=1)
        if not top:
            return self.similarity(text, query)
        return top[0][1]


def get_semantic_engine(model_name: str = DEFAULT_MODEL) -> SemanticEngine:
    """Get or create a cached SemanticEngine for the given model."""
    with _LOCK:
        if model_name not in _engines:
            _engines[model_name] = SemanticEngine(model_name)
        return _engines[model_name]
