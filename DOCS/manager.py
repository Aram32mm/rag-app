"""
manager.py — Embedding Manager

Purpose
-------
Provides a manager class for generating high-quality sentence embeddings
using HuggingFace transformer models.

Key Responsibilities
--------------------
- Load model and tokenizer from HuggingFace.
- Handle device placement (CPU, CUDA, MPS).
- Apply mask-aware mean pooling (ignoring padding).
- Perform L2 normalization for cosine similarity.
- Support batch embedding and optional in-memory caching.

Dependencies
------------
- torch
- transformers (AutoTokenizer, AutoModel)
- logging
"""

from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages transformer-based embedding models and generates embeddings.

    Features
    --------
    - Attention-mask-aware mean pooling (ignores padding).
    - L2-normalization (unit vectors for cosine similarity).
    - Batch embedding for speed.
    - In-memory caching to avoid recomputation.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        """
        Initialize the EmbeddingManager.

        Args
        ----
        model_name : str
            HuggingFace model name or local path.
        device : str, optional
            Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = self._get_device() if device is None else device
        self.model = None
        self.tokenizer = None
        self._embedding_cache = {}
        logger.info("EmbeddingManager initialized with model '%s' on device '%s'.",
                    self.model_name, self.device)

    def _get_device(self) -> str:
        """Detect and return the best available device for inference."""
        if torch.backends.mps.is_available():
            logger.debug("Using MPS device.")
            return "mps"
        elif torch.cuda.is_available():
            logger.debug("Using CUDA device.")
            return "cuda"
        else:
            logger.debug("Using CPU device.")
            return "cpu"

    def load_model(self):
        """Load the HuggingFace transformer model and tokenizer onto the device."""
        logger.info("Loading model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-mask-aware mean pooling.

        Args
        ----
        model_output : Any
            Output from transformer model.
        attention_mask : torch.Tensor
            Mask indicating non-padding tokens.

        Returns
        -------
        torch.Tensor
            Mean-pooled embeddings of shape [batch, hidden_dim].
        """
        token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden_dim]
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        sum_embeddings = (token_embeddings * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)  # prevent division by zero
        return sum_embeddings / token_counts

    def generate_embeddings(self, texts: Union[str, List[str]],
                             batch_size: int = 32,
                             use_cache: bool = True) -> torch.Tensor:
        """
        Generate embeddings for text(s).

        Args
        ----
        texts : str | List[str]
            Input text(s).
        batch_size : int
            Batch size for model inference.
        use_cache : bool
            Whether to use cache.

        Returns
        -------
        numpy.ndarray
            L2-normalized embeddings of shape [num_texts, hidden_dim].
        """
        if self.model is None:
            self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        logger.info("Generating embeddings for %d texts (batch_size=%d).",
                    len(texts), batch_size)
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug("Processing batch %d: %s", (i // batch_size) + 1, batch)

            uncached = []
            for t in batch:
                if use_cache and t in self._embedding_cache:
                    embeddings.append(self._embedding_cache[t])
                else:
                    uncached.append(t)

            if uncached:
                logger.info("Encoding %d uncached texts.", len(uncached))
                encoded = self.tokenizer(
                    uncached, padding=True, truncation=True,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded)
                    pooled = self._mean_pooling(model_output, encoded["attention_mask"])
                    normalized = F.normalize(pooled, p=2, dim=1)  # L2 normalize

                for j, t in enumerate(uncached):
                    vec = normalized[j].cpu()
                    if use_cache:
                        self._embedding_cache[t] = vec
                    embeddings.append(vec)

        logger.info("Embeddings generated successfully.")
        return torch.stack(embeddings).numpy()
