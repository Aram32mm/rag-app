"""
manager.py — SQLite database utilities

Purpose
-------
Create and access the rules table, optionally ingesting from CSV, and provide
helpers to fetch or upsert rules.

Key Responsibilities
--------------------
- Initialize table schema; optional CSV import on first run.
- Fetch all rules (`get_rules`).
- Upsert a single rule (`upsert_rule`).

Dependencies
------------
- sqlite3, pandas (CSV import)
- logging, datetime, typing
"""


from typing import List, Union, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    EmbeddingManager
    ----------------
    Loads and manages transformer-based embedding models,
    and generates embeddings for input text.

    Key features:
    - Attention-mask-aware mean pooling (ignores padding).
    - L2-normalizes embeddings (unit vectors).
    - Supports batch processing for efficiency.
    - Caches embeddings to avoid recomputation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the EmbeddingManager.

        Args:
            model_name (str): HuggingFace model name or local path.
            device (str, optional): Device to use ('cpu', 'cuda', 'mps').
                                    If None, auto-detected.
        """
        self.model_name = model_name
        self.device = self._get_device() if device is None else device
        self.model = None
        self.tokenizer = None
        self._embedding_cache = {}  # in-memory cache: text -> embedding
        logger.info(
            f"EmbeddingManager initialized with model '{self.model_name}' on device '{self.device}'."
        )

    # -----------------------------
    # Device handling
    # -----------------------------
    def _get_device(self) -> str:
        """Detect and return the best available device for inference."""
        if torch.backends.mps.is_available():
            logger.debug("MPS device available and selected.")
            return "mps"
        elif torch.cuda.is_available():
            logger.debug("CUDA device available and selected.")
            return "cuda"
        else:
            logger.debug("Defaulting to CPU device.")
            return "cpu"

    # -----------------------------
    # Model loading
    # -----------------------------
    def load_model(self):
        """Load the HuggingFace transformer model and tokenizer onto the device."""
        logger.info(f"Loading transformer model '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise
        logger.info("Transformer model loaded successfully.")

    # -----------------------------
    # Pooling
    # -----------------------------
    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-mask-aware mean pooling.

        Args:
            model_output: Output from transformer model.
            attention_mask: Mask indicating non-padding tokens.

        Returns:
            torch.Tensor: Mean-pooled embeddings of shape [batch, hidden_dim].
        """
        token_embeddings = model_output.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Expand mask so it can be multiplied with token embeddings
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)

        # Weighted sum of embeddings
        sum_embeddings = (token_embeddings * mask).sum(dim=1)

        # Divide by number of non-padding tokens (clamped to avoid div/0)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)

        return sum_embeddings / token_counts

    # -----------------------------
    # Embedding generation
    # -----------------------------
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate embeddings for text(s).

        Args:
            texts (str or List[str]): Input text(s).
            batch_size (int): Batch size for model inference.
            use_cache (bool): Whether to use in-memory cache.

        Returns:
            np.ndarray: L2-normalized embeddings of shape [num_texts, hidden_dim].
        """
        if self.model is None:
            self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            logger.warning("generate_embeddings called with empty text list.")
            return torch.empty((0, self.model.config.hidden_size))

        logger.info(
            f"Generating embeddings for {len(texts)} text(s) with batch size {batch_size}."
        )
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}/{-(-len(texts)//batch_size)}.")

            uncached, uncached_idx = [], []

            # Check cache for each text in batch
            for j, t in enumerate(batch):
                if use_cache and t in self._embedding_cache:
                    embeddings.append(self._embedding_cache[t])
                else:
                    uncached.append(t)
                    uncached_idx.append(j)

            # Encode only uncached texts
            if uncached:
                logger.info(f"Encoding {len(uncached)} uncached text(s).")
                encoded = self.tokenizer(
                    uncached, padding=True, truncation=True, return_tensors="pt"
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
