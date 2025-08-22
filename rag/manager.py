"""
manager.py

Provides a manager class for generating high-quality sentence embeddings
using HuggingFace transformer models. Handles:
- Model & tokenizer loading
- Device placement (CPU, CUDA, MPS)
- Mask-aware mean pooling (ignoring padding)
- L2 normalization (for cosine similarity)
- Optional in-memory caching

This ensures embeddings are consistent, comparable, and ready for use in retrieval.
"""

from typing import List, Union, Optional
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel


class EmbeddingManager:
    """
    EmbeddingManager
    ----------------
    Loads and manages transformer-based embedding models, and generates
    embeddings for input text.

    Key features:
    - Uses attention-mask-aware mean pooling (ignores padding).
    - L2-normalizes embeddings (unit vectors).
    - Supports batch embedding for speed.
    - Supports caching to avoid recomputation.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        """
        Initialize the EmbeddingManager.

        Args:
            model_name (str): HuggingFace model name or local path.
            device (str, optional): Device to use ('cpu', 'cuda', 'mps').
                                    Auto-detected if None.
        """
        self.model_name = model_name
        self.device = self._get_device() if device is None else device
        self.model = None
        self.tokenizer = None
        self._embedding_cache = {}

    # -----------------------------
    # Device detection
    # -----------------------------
    def _get_device(self) -> str:
        """Detect and return the best available device for inference."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    # -----------------------------
    # Model loading
    # -----------------------------
    def load_model(self):
        """Load the HuggingFace transformer model and tokenizer onto the device."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    # -----------------------------
    # Pooling
    # -----------------------------
    def _mean_pooling(self, model_output, attention_mask):
        """
        Apply attention-mask-aware mean pooling.

        Args:
            model_output: Output from transformer model.
            attention_mask: Mask indicating non-padding tokens.

        Returns:
            torch.Tensor: Mean-pooled embeddings of shape [batch, hidden_dim].
        """
        # return sum_embeddings / sum_mask
        token_embeddings = model_output.last_hidden_state  # [B, T, H]
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)  # [B, T, 1]
        sum_embeddings = (token_embeddings * mask).sum(dim=1)                 # [B, H]
        # Count of real tokens per sequence (broadcastable to [B, H])
        token_counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B, 1]
        return sum_embeddings / token_counts    

    # -----------------------------
    # Embedding generation
    # -----------------------------
    def generate_embeddings(self, texts: Union[str, List[str]], batch_size: int = 32,
                             use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts (str or List[str]): Input text(s).
            batch_size (int): Batch size for model inference.
            use_cache (bool): Whether to use cache.

        Returns:
            np.ndarray: L2-normalized embeddings of shape [num_texts, hidden_dim].
        """
        if self.model is None:
            self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            uncached = []
            uncached_idx = []

            # Check cache
            for j, t in enumerate(batch):
                if use_cache and t in self._embedding_cache:
                    embeddings.append(self._embedding_cache[t])
                else:
                    uncached.append(t)
                    uncached_idx.append(j)

            if uncached:
                encoded = self.tokenizer(
                    uncached,
                    padding=True,
                    truncation=True,
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

        # Convert list[Tensor] → ndarray [N, D]
        return torch.stack(embeddings).numpy()


# -----------------------------
# Sanity check when run directly
# -----------------------------
if __name__ == "__main__":
    import numpy as np

    manager = EmbeddingManager(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Bananas are yellow and sweet."
    ]

    embeddings = manager.generate_embeddings(texts)

    print("\n=== EmbeddingManager Sanity Check ===")
    for i, emb in enumerate(embeddings):
        norm = np.linalg.norm(emb)
        print(f"[{i}] shape={emb.shape}, norm={norm:.4f}")
    print("✅ Embeddings generated and normalized.")
