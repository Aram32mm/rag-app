# base.py

from enum import Enum
from dataclasses import dataclass

class SearchMode(str, Enum):
    """Available retrieval modes."""
    SEMANTIC = "semantic"
    BM25 = "bm25"
    FUZZY = "fuzzy"
    HYBRID = "hybrid"


@dataclass
class SearchConfig:
    """Configuration for retrieval scoring and thresholds."""
    semantic_weight: float = 0.8
    bm25_weight: float = 0.1
    fuzzy_weight: float = 0.1
    min_similarity: float = 0.0
    enable_reranking: bool = False
    top_k: int = 5
