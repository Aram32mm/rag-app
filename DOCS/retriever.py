"""
retriever.py — Hybrid rule retriever

Purpose
-------
Provide semantic, keyword (BM25), and fuzzy scoring, filterable by facets,
with per-signal normalization and hybrid weighted ranking.

Key Responsibilities
--------------------
- Build/maintain embedding and BM25 indices.
- Expose `search_rules` with facet filters and search modes.
- Compute semantic/BM25/fuzzy scores and normalize per prompt.
- Combine scores with convex weights; optional reranking.
- Supply filter option values for the UI.

Dependencies
------------
- numpy, rank_bm25, fuzzywuzzy
- rag.embeddings.manager.EmbeddingManager
- rag.embeddings.index.EmbeddingIndex
- rag.search.config (SearchConfig, SearchMode)
- rag.search.data (RuleDataLoader)
- db.manager.DatabaseManager
"""


from typing import List, Dict, Optional
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi

# Project-local imports
from .config import SearchConfig, SearchMode
from .data import RuleDataLoader
from ..embeddings.manager import EmbeddingManager
from ..embeddings.index import EmbeddingIndex
from db.manager import DatabaseManager

logger = logging.getLogger(__name__)


class RuleRetriever:
    """
    Core retriever class that supports:
    - Semantic retrieval (via embeddings)
    - BM25 keyword search
    - Fuzzy string matching
    - Hybrid convex-weight scoring
    - Optional filter constraints
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[SearchConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        # --- Core setup ---
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.config = config or SearchConfig()
        self.data_loader = RuleDataLoader(db_manager)

        # All available rules in memory
        self.rules: List[dict] = self.data_loader.rules

        # Search indices
        self.embedding_index: Optional[EmbeddingIndex] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self._bm25_keywords_len: int = 0

        # Cached filter options (country, type, etc.)
        self.filter_options: Optional[Dict[str, List[str]]] = None

        # Fail fast if no rules
        if not self.rules:
            logger.error("No rules loaded from the database. Please check your data source.")
            raise ValueError("No rules loaded from the database. Please check your data source.")

        # Build indices and filters
        self._build_indices()
        self._build_filter_options()

        logger.info("RuleRetriever initialised with %d rules", len(self.rules))

    # -----------------------------
    # Index building
    # -----------------------------
    def _build_indices(self):
        """Build semantic and BM25 indices in parallel."""

        def build_embedding_index():
            logger.info("[Retriever] Building embedding index...")
            self.embedding_index = EmbeddingIndex(self.embedding_manager)
            self.embedding_index.add_rules(self.rules)
            logger.info("[Retriever] Embedding index built successfully with %d rules.", len(self.rules))

        def build_bm25_index():
            logger.info("[Retriever] Building BM25 index...")
            corpus = [r.get("keywords", "") for r in self.rules]
            tokenized = [kw.replace(",", " ").split() if kw else [] for kw in corpus]

            if not tokenized or all(len(toks) == 0 for toks in tokenized):
                self.bm25_index = None
                self._bm25_keywords_len = 0
                logger.warning("[Retriever] BM25 index not built (empty or invalid keywords).")
            else:
                self.bm25_index = BM25Okapi(tokenized)
                self._bm25_keywords_len = len(tokenized)
                logger.info("[Retriever] BM25 index built successfully with %d keyword docs.", self._bm25_keywords_len)

        # Build both indices in parallel
        with ThreadPoolExecutor() as exe:
            futures = [exe.submit(build_embedding_index), exe.submit(build_bm25_index)]
            for f in futures:
                f.result()  # block until complete

    def _build_filter_options(self) -> Dict[str, List[str]]:
        """Extract unique filter options (country, rule_type, etc.) from rules."""

        def flatten_unique(field):
            tags = set()
            for r in self.rules:
                vals = r.get(field, [])
                if isinstance(vals, (list, set, tuple)):
                    tags.update(vals)
                elif vals:
                    tags.add(vals)
            return sorted(tags)

        self.filter_options = {
            "rule_type": flatten_unique("rule_type"),
            "country": flatten_unique("country"),
            "business_type": flatten_unique("business_type"),
            "party_agent": flatten_unique("party_agent"),
        }
        logger.info("[Retriever] Filter options built: %s", self.filter_options)
        return self.filter_options

    # -----------------------------
    # Public API
    # -----------------------------
    def search_rules(
        self,
        query: Optional[str] = None,
        rule_type: Optional[List[str]] = None,
        country: Optional[List[str]] = None,
        business_type: Optional[List[str]] = None,
        party_agent: Optional[List[str]] = None,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
    ) -> List[dict]:
        """
        Main search entrypoint used by the app.

        Args:
            query: User query string
            rule_type, country, business_type, party_agent: Filter options
            mode: SearchMode (semantic, bm25, fuzzy, hybrid)
            top_k: Max results to return

        Returns:
            List of rule dicts with added 'search_score'
        """

        # Apply filters
        rules = self._apply_filters(
            self.rules, rule_type=rule_type, country=country, business_type=business_type, party_agent=party_agent
        )
        if not rules:
            logger.info("[Retriever] No rules matched after applying filters.")
            return []

        if not query:
            logger.info("[Retriever] No query provided, returning first %d rules.", top_k)
            return rules[:top_k]

        # Select retrieval mode
        if mode == SearchMode.SEMANTIC:
            scores = self._semantic_scores(query, rules)
        elif mode == SearchMode.KEYWORD:
            scores = self._bm25_scores(query, rules)
        elif mode == SearchMode.FUZZY:
            scores = self._fuzzy_scores(query, rules)
        else:
            scores = self._hybrid_scores(query, rules)

        # Attach scores to rules
        scores_vec = [scores.get(r["rule_id"], 0.0) for r in rules]
        results = [dict(r, search_score=s) for r, s in zip(rules, scores_vec) if s >= self.config.min_similarity]

        # Sort by score
        results.sort(key=lambda r: r["search_score"], reverse=True)

        # Optional reranking
        if self.config.enable_reranking and len(results) > top_k:
            logger.info("[Retriever] Applying reranking stage.")
            results = self._rerank_results(query, results, top_k)

        logger.debug("[Retriever] Returning %d results for query='%s'", len(results[:top_k]), query)
        return results[:top_k]

    # -----------------------------
    # Filtering
    # -----------------------------
    @staticmethod
    def _to_list(val) -> List[str]:
        """Ensure filter values are always a list."""
        if val is None:
            return []
        if isinstance(val, (list, set, tuple)):
            return list(val)
        return [val]

    def _apply_filters(
        self,
        rules: List[dict],
        rule_type: Optional[List[str]] = None,
        country: Optional[List[str]] = None,
        business_type: Optional[List[str]] = None,
        party_agent: Optional[List[str]] = None,
    ) -> List[dict]:
        """Filter rules by provided categorical values."""
        filters = {
            "rule_type": self._to_list(rule_type),
            "country": self._to_list(country),
            "business_type": self._to_list(business_type),
            "party_agent": self._to_list(party_agent),
        }
        out = rules
        for field, selected in filters.items():
            if selected:
                sel = set(selected)
                out = [r for r in out if set(r.get(field, [])) & sel]
        return out

    # -----------------------------
    # Per-signal scoring
    # -----------------------------
    def _semantic_scores(self, query: str, rules: List[dict]) -> Dict[str, float]:
        """Return semantic similarity scores."""
        if self.embedding_index is None:
            logger.warning("[Retriever] Semantic index not available.")
            return {r["rule_id"]: 0.0 for r in rules}

        results = self.embedding_index.search(query, top_k=len(rules))
        by_id = {r["rule_id"]: float(score) for r, score in results}
        return {r["rule_id"]: by_id.get(r["rule_id"], 0.0) for r in rules}

    def _bm25_scores(self, query: str, rules: List[dict]) -> Dict[str, float]:
        """Return BM25 keyword match scores."""
        if self.bm25_index is None or self._bm25_keywords_len == 0:
            logger.warning("[Retriever] BM25 index not available.")
            return {r["rule_id"]: 0.0 for r in rules}

        q_tokens = query.replace(",", " ").split()
        scores = self.bm25_index.get_scores(q_tokens)
        id_to_pos = {r["rule_id"]: i for i, r in enumerate(self.rules)}

        out = {}
        for r in rules:
            pos = id_to_pos.get(r["rule_id"], None)
            out[r["rule_id"]] = float(scores[pos]) if pos is not None and 0 <= pos < len(scores) else 0.0
        return out

    def _fuzzy_scores(self, query: str, rules: List[dict]) -> Dict[str, float]:
        """Return fuzzy string match scores."""
        out = {}
        for r in rules:
            name = r.get("rule_name", "") or ""
            out[r["rule_id"]] = float(fuzz.partial_ratio(query.lower(), name.lower())) / 100.0
        return out

    # -----------------------------
    # Normalization & hybrid
    # -----------------------------
    @staticmethod
    def _normalize_per_prompt(scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0, 1] relative to max."""
        if not scores:
            return scores
        mx = max(scores.values())
        if mx <= 0:
            return {k: 0.0 for k in scores}
        return {k: v / mx for k, v in scores.items()}

    def _hybrid_scores(self, query: str, rules: List[dict]) -> Dict[str, float]:
        """Weighted convex combination of semantic, BM25, and fuzzy scores."""
        sem = self._normalize_per_prompt(self._semantic_scores(query, rules))
        bm25 = self._normalize_per_prompt(self._bm25_scores(query, rules))
        fz = self._normalize_per_prompt(self._fuzzy_scores(query, rules))

        # Normalize weights
        w_sem, w_kw, w_fz = (
            max(self.config.semantic_weight, 0.0),
            max(self.config.bm25_weight, 0.0),
            max(self.config.fuzzy_weight, 0.0),
        )
        sw = w_sem + w_kw + w_fz
        if sw == 0:
            w_sem = w_kw = w_fz = 1.0 / 3.0
            sw = 1.0

        w_sem, w_kw, w_fz = w_sem / sw, w_kw / sw, w_fz / sw

        out = {}
        for r in rules:
            rid = r["rule_id"]
            out[rid] = w_sem * sem.get(rid, 0.0) + w_kw * bm25.get(rid, 0.0) + w_fz * fz.get(rid, 0.0)
        return out

    # -----------------------------
    # Candidate pooling (evaluation)
    # -----------------------------
    def candidate_pool(self, query: str, k_each: int = 20) -> List[dict]:
        """Return deduplicated candidate set as union of top-k from semantic, BM25, and fuzzy."""
        sem_ids = self._top_k_ids(self._semantic_scores(query, self.rules), k_each)
        bm_ids = self._top_k_ids(self._bm25_scores(query, self.rules), k_each)
        fz_ids = self._top_k_ids(self._fuzzy_scores(query, self.rules), k_each)

        pool_ids = list(dict.fromkeys(sem_ids + bm_ids + fz_ids))  # dedupe while preserving order
        id_to_rule = {r["rule_id"]: r for r in self.rules}
        return [id_to_rule[rid] for rid in pool_ids if rid in id_to_rule]

    @staticmethod
    def _top_k_ids(score_dict: Dict[str, float], k: int) -> List[str]:
        return [rid for rid, _ in sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    # -----------------------------
    # Reranking
    # -----------------------------
    def _rerank_results(self, query: str, results: List[dict], top_k: int) -> List[dict]:
        """Optional reranking stage (currently pass-through)."""
        logger.debug("[Retriever] Reranking placeholder invoked.")
        return results[:top_k]
