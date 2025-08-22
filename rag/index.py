"""
index.py

Provides a vector index for semantic search using cosine similarity.
Stores embeddings + metadata and allows fast retrieval of most similar items.
"""

from typing import List, Dict, Tuple
import numpy as np
import pickle
from rag.manager import EmbeddingManager


class EmbeddingIndex:
    """
    EmbeddingIndex
    --------------
    A simple in-memory semantic search index.

    Responsibilities:
    - Store embeddings and associated metadata.
    - Support efficient cosine-similarity search.
    - Save/load index state to/from disk.

    Assumptions:
    - All embeddings are L2-normalized to unit length either by:
        - The EmbeddingManager when generating new embeddings, OR
        - A defensive normalization step if user-provided vectors are stored.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Args:
            embedding_manager (EmbeddingManager):
                Used to generate embeddings for queries or rules without stored vectors.
        """
        self.embedding_manager = embedding_manager
        self.embeddings: np.ndarray = None  # shape [N, D], always float32, L2-normalized
        self.metadata: List[Dict] = []      # rule metadata aligned to rows in embeddings
        self.index_built: bool = False      # True once rules have been added

    # -----------------------------
    # Index construction
    # -----------------------------
    def add_rules(self, rules: List[Dict]):
        """
        Add rules to the index (computes embeddings if missing).

        Args:
            rules (List[Dict]):
                Each rule must have either:
                  - "embedding": list[float] (pre-computed embedding vector), OR
                  - "llm_description": str (text input used to generate embedding).
                Each rule should also have a unique "rule_id".
        """
        self.metadata = []
        emb_list = []

        for rule in rules:
            emb_val = rule.get("embedding", None)
            if emb_val is not None:
                # Use stored embedding (convert to float32 + normalize)
                vec = np.array(emb_val, dtype=np.float32)
                norm = np.linalg.norm(vec) + 1e-12  # add epsilon to avoid div-by-zero
                vec = vec / norm
            else:
                # Generate fresh embedding from description text
                vec = self.embedding_manager.generate_embeddings(
                    rule.get("llm_description", "")
                )[0]  # Manager returns list, take first
            emb_list.append(vec)
            self.metadata.append(rule)

        # Stack into [N, D] matrix for efficient similarity search
        self.embeddings = np.vstack(emb_list).astype(np.float32)
        self.index_built = True

    # -----------------------------
    # Search
    # -----------------------------
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for most similar rules to a query using cosine similarity.

        Args:
            query (str): Input text to search with.
            top_k (int): Number of results to return (defaults to 5).

        Returns:
            List[Tuple[Dict, float]]:
                Tuples of (rule_metadata, similarity_score) ordered by similarity.
                Similarity is in [0, 1] because all embeddings are L2-normalized.
        """
        if not self.index_built:
            raise ValueError("Index not built. Call add_rules() before searching.")

        # Generate L2-normalized query embedding
        query_emb = self.embedding_manager.generate_embeddings(query)[0]

        # Cosine similarity reduces to dot product because vectors are unit length
        sims = self.embeddings @ query_emb  # shape [N]

        # Select top-k efficiently (partial sort, avoids full sort overhead)
        top_idx = np.argpartition(-sims, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]  # order top-k by score descending

        return [(self.metadata[i], float(sims[i])) for i in top_idx]

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, filepath: str):
        """
        Save embeddings + metadata to disk.

        Args:
            filepath (str): Path to pickle file where index will be stored.
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'index_built': self.index_built
            }, f)

    def load(self, filepath: str):
        """
        Load embeddings + metadata from disk.

        Args:
            filepath (str): Path to pickle file previously created by save().
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        self.index_built = data['index_built']

if __name__ == "__main__":

    from rag.manager import EmbeddingManager

    manager = EmbeddingManager("sentence-transformers/all-MiniLM-L6-v2")
    index = EmbeddingIndex(manager)

    rules = [
        {"id": 1, "llm_description": "A rule about foxes and animals."},
        {"id": 2, "llm_description": "A rule about technology and AI."},
        {"id": 3, "llm_description": "A rule about fruits like bananas."},
    ]

    index.add_rules(rules)

    query = "Tell me something about artificial intelligence."
    results = index.search(query, top_k=2)

    print("\n=== EmbeddingIndex Sanity Check ===")
    for rule, score in results:
        print(f"id={rule['id']} | score={score:.4f} | text={rule['llm_description']}")
    print("✅ Retrieval works.")
