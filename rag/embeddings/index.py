"""
embedding_index.py

Provides a vector index for semantic search using cosine similarity.
Stores embeddings + metadata and allows fast retrieval of most similar items.
"""

from typing import List, Dict, Tuple
import numpy as np
import pickle
from embedding_manager import EmbeddingManager


class EmbeddingIndex:
    """
    EmbeddingIndex
    --------------
    A simple in-memory semantic search index.

    Responsibilities:
    - Store embeddings and associated metadata.
    - Support efficient cosine-similarity search.
    - Save/load index state to/from disk.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.embeddings = None  # shape [N, D], L2-normalized
        self.metadata = []      # list of rule dicts
        self.index_built = False

    def add_rules(self, rules: List[Dict]):
        """
        Add rules to the index (computes embeddings if missing).

        Args:
            rules (List[Dict]): Each rule should have at least
                                'llm_description' or 'embedding'.
        """
        self.metadata = []
        emb_list = []

        for rule in rules:
            # Use stored embedding if present, else compute
            emb_val = rule.get("embedding", None)
            if emb_val is not None:
                vec = np.array(emb_val, dtype=np.float32)
                # defensive L2 normalization
                norm = np.linalg.norm(vec) + 1e-12
                vec = vec / norm
            else:
                vec = self.embedding_manager.generate_embeddings(
                    rule.get("llm_description", "")
                )[0]
            emb_list.append(vec)
            self.metadata.append(rule)

        self.embeddings = np.vstack(emb_list).astype(np.float32)
        self.index_built = True

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for most similar rules to a query.

        Args:
            query (str): Input text to search with.
            top_k (int): Number of results to return.

        Returns:
            List[(rule_dict, similarity_score)]
        """
        if not self.index_built:
            raise ValueError("Index not built. Add rules first.")

        # embed query (already L2 normalized by manager)
        query_emb = self.embedding_manager.generate_embeddings(query)[0]

        # cosine similarity = dot product since embeddings are unit length
        sims = self.embeddings @ query_emb

        # get top-k results efficiently
        top_idx = np.argpartition(-sims, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        return [(self.metadata[i], float(sims[i])) for i in top_idx]

    def save(self, filepath: str):
        """Save embeddings + metadata to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'index_built': self.index_built
            }, f)

    def load(self, filepath: str):
        """Load embeddings + metadata from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        self.index_built = data['index_built']
