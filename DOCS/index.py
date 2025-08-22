"""
index.py — In-memory embedding index

Purpose
-------
Store L2-normalized embeddings with metadata and support fast cosine-similarity
search, plus simple save/load.

Key Responsibilities
--------------------
- Add rules (use precomputed or generated embeddings).
- Execute top-k cosine search for a query.
- Persist/restore index state (pickle).

Dependencies
------------
- numpy, pickle
- rag.embeddings.manager.EmbeddingManager
- logging
"""


from typing import List, Dict, Tuple
import numpy as np
import pickle
import logging

from .manager import EmbeddingManager

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """
    EmbeddingIndex
    --------------
    A simple in-memory semantic search index.

    Responsibilities:
    - Store embeddings (L2-normalized) and associated rule metadata.
    - Support efficient cosine similarity search.
    - Persist / restore index state from disk.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.embeddings: np.ndarray = None   # shape [N, D], L2-normalized
        self.metadata: List[Dict] = []       # list of rule dicts
        self.index_built: bool = False
        logger.info("EmbeddingIndex initialized.")

    # -----------------------------
    # Index construction
    # -----------------------------
    def add_rules(self, rules: List[Dict]):
        """
        Add rules to the index (computes embeddings if missing).

        Args:
            rules (List[Dict]): Each rule should contain either:
                                - 'embedding': precomputed vector, OR
                                - 'llm_description': text to embed.
        """
        if not rules:
            logger.warning("add_rules called with empty rule list.")
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.metadata = []
            self.index_built = True
            return

        logger.info(f"Adding {len(rules)} rules to the embedding index.")

        emb_list = []
        self.metadata = []

        for rule in rules:
            emb_val = rule.get("embedding", None)

            if emb_val is not None:
                # Convert stored embedding to np.array + normalize defensively
                vec = np.array(emb_val, dtype=np.float32)
                norm = np.linalg.norm(vec) + 1e-12
                vec = vec / norm
            else:
                # Compute embedding from description
                desc = rule.get("llm_description", "")
                if not desc:
                    logger.debug(f"Rule {rule.get('rule_id')} missing description; using zero vector.")
                    vec = np.zeros((self.embedding_manager.dim,), dtype=np.float32)
                else:
                    vec = self.embedding_manager.generate_embeddings(desc)[0]

            emb_list.append(vec)
            self.metadata.append(rule)

        self.embeddings = np.vstack(emb_list).astype(np.float32)
        self.index_built = True
        logger.info(f"Embedding index built successfully with {len(self.metadata)} rules.")

    # -----------------------------
    # Search
    # -----------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve the most similar rules to a given query.

        Args:
            query (str): Input text query.
            top_k (int): Number of results to return.

        Returns:
            List of (rule_dict, similarity_score), sorted by score.
        """
        if not self.index_built:
            logger.error("Search failed: index not built. Call add_rules first.")
            raise ValueError("Index not built. Add rules first.")

        if self.embeddings is None or self.embeddings.shape[0] == 0:
            logger.warning("Search requested on empty embedding index.")
            return []

        logger.debug(f"Searching embedding index: query='{query}', top_k={top_k}")

        # Embed query (EmbeddingManager ensures normalization)
        query_emb = self.embedding_manager.generate_embeddings(query)[0]

        # Cosine similarity via dot product (since vectors are unit length)
        sims = self.embeddings @ query_emb
        N = sims.shape[0]

        # Clip top_k to valid range
        k = max(1, min(top_k, N))

        # Partial sort for efficiency, then full sort on subset
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = [(self.metadata[i], float(sims[i])) for i in top_idx]
        logger.info(f"Search complete: {len(results)} results for query '{query}'.")
        return results

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, filepath: str):
        """Persist embeddings + metadata to disk."""
        if not self.index_built:
            logger.warning("Attempted to save an unbuilt index.")
        logger.info(f"Saving embedding index to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "embeddings": self.embeddings,
                    "metadata": self.metadata,
                    "index_built": self.index_built,
                },
                f,
            )
        logger.info("Embedding index saved successfully.")

    def load(self, filepath: str):
        """Load embeddings + metadata from disk."""
        logger.info(f"Loading embedding index from {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data.get("embeddings", None)
        self.metadata = data.get("metadata", [])
        self.index_built = data.get("index_built", False)
        logger.info(f"Embedding index loaded. Contains {len(self.metadata)} rules.")
