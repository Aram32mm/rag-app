
"""
Improved RuleRetriever with:
- Explicit per-signal scores (semantic/BM25/fuzzy)
- Per-prompt normalization
- Convex hybrid scoring (non-negative weights that sum to 1)
- Candidate pool utility (union of top-k from each signal)
- Robust BM25: supports "keywords" or legacy "keywoards"
- Fuzzy on rule_name (falls back to name)

Assumes project has:
- SearchConfig with fields: semantic_weight, bm25_weight, fuzzy_weight, min_similarity, enable_reranking
- SearchMode enum with SEMANTIC, KEYWORD, FUZZY, HYBRID
- RuleDataLoader -> loads rules list[dict] with keys like: rule_id, rule_name, llm_description, keywords/keywoards, ...
- EmbeddingManager + EmbeddingIndex
- DatabaseManager
"""

from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor

from fuzzywuzzy import fuzz
from rank_bm25 import BM25Okapi

from .base import SearchConfig, SearchMode
from .data import RuleDataLoader
from ..embeddings.embeddings_manager import EmbeddingManager
from ..embeddings.embedding_index import EmbeddingIndex
from db.manager import DatabaseManager

logger = logging.getLogger(__name__)

class RuleRetriever:
    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        config: Optional[SearchConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.config = config or SearchConfig()
        self.data_loader = RuleDataLoader(db_manager)
        self.rules: List[dict] = self.data_loader.rules
        self.embedding_index: Optional[EmbeddingIndex] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self._bm25_keywords_len: int = 0
        self.filter_options: Optional[Dict[str, List[str]]] = None

        if not self.rules:
            logger.error("No rules loaded from the database. Please check your data source.")
            raise ValueError("No rules loaded from the database. Please check your data source.")

        self._build_indices()
        self._build_filter_options()

    # -----------------------------
    # Index building
    # -----------------------------
    def _build_indices(self):
        def build_embedding_index():
            logger.info("Building embedding index...")
            self.embedding_index = EmbeddingIndex(self.embedding_manager)
            # The EmbeddingIndex should internally use 'llm_description' (or similar) for vectors.
            self.embedding_index.add_rules(self.rules)
            logger.info("Embedding index built successfully.")

        def build_bm25_index():
            logger.info("Building BM25 index...")
            # Support both "keywords" and legacy misspelling "keywoards"
            corpus = [
                (r.get("keywords") if r.get("keywords") is not None else r.get("keywoards", ""))
                for r in self.rules
            ]
            tokenized = [kw.replace(",", " ").split() if kw else [] for kw in corpus]
            if not tokenized or all(len(toks) == 0 for toks in tokenized):
                self.bm25_index = None
                self._bm25_keywords_len = 0
                logger.warning("BM25 index not built due to empty/invalid keywords/keywoards.")
            else:
                self.bm25_index = BM25Okapi(tokenized)
                self._bm25_keywords_len = len(tokenized)
                logger.info("BM25 index built successfully.")

        with ThreadPoolExecutor() as exe:
            futures = [exe.submit(build_embedding_index), exe.submit(build_bm25_index)]
            for f in futures:
                f.result()

    def _build_filter_options(self) -> Dict[str, List[str]]:
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
        rules = self._apply_filters(
            self.rules, rule_type=rule_type, country=country, business_type=business_type, party_agent=party_agent
        )
        if not rules:
            return []

        if not query:
            return rules[:top_k]

        if mode == SearchMode.SEMANTIC:
            scores = self._semantic_scores(query, rules)
        elif mode == SearchMode.KEYWORD:
            scores = self._bm25_scores(query, rules)
        elif mode == SearchMode.FUZZY:
            scores = self._fuzzy_scores(query, rules)
        else:
            scores = self._hybrid_scores(query, rules)

        scores_vec = [scores.get(r["rule_id"], 0.0) for r in rules]
        results = [dict(r, search_score=s) for r, s in zip(rules, scores_vec) if s >= self.config.min_similarity]
        results.sort(key=lambda r: r["search_score"], reverse=True)

        if self.config.enable_reranking and len(results) > top_k:
            results = self._rerank_results(query, results, top_k)

        return results[:top_k]

    # -----------------------------
    # Filtering
    # -----------------------------
    @staticmethod
    def _to_list(val) -> List[str]:
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
    # Per-signal scoring (dict: rule_id -> raw score)
    # -----------------------------
    def _semantic_scores(self, query: str, rules: List[dict]) -> Dict[str, float]:
        if self.embedding_index is None:
            return {r["rule_id"]: 0.0 for r in rules}
        results = self.embedding_index.search(query, top_k=len(rules))
        by_id = {r["rule_id"]: float(score) for r, score in results}
        return {r["rule_id"]: by_id.get(r["rule_id"], 0.0) for r in rules}

    def _bm25_scores(self, query: str, rules: List[dict]) -> Dict[str, float]:
        if self.bm25_index is None or self._bm25_keywords_len == 0:
            return {r["rule_id"]: 0.0 for r in rules}
        q_tokens = query.replace(",", " ").split()
        scores = self.bm25_index.get_scores(q_tokens)
        id_to_pos = {r["rule_id"]: i for i, r in enumerate(self.rules)}
        out = {}
        for r in rules:
            pos = id_to_pos.get(r["rule_id"])
            out[r["rule_id"]] = float(scores[pos]) if pos is not None else 0.0
        return out

    def _fuzzy_scores(self, query: str, rules: List[dict]) -> Dict[str, float]:
        out = {}
        for r in rules:
            name = (r.get("rule_name") or r.get("name") or "")
            out[r["rule_id"]] = float(fuzz.partial_ratio(query.lower(), name.lower())) / 100.0
        return out

    # -----------------------------
    # Normalization & hybrid
    # -----------------------------
    @staticmethod
    def _normalize_per_prompt(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return scores
        mx = max(scores.values())
        if mx <= 0:
            return {k: 0.0 for k in scores}
        return {k: v / mx for k, v in scores.items()}

    def _hybrid_scores(self, query: str, rules: List[dict]) -> Dict[str, float]:
        sem = self._normalize_per_prompt(self._semantic_scores(query, rules))
        bm25 = self._normalize_per_prompt(self._bm25_scores(query, rules))
        fz = self._normalize_per_prompt(self._fuzzy_scores(query, rules))

        w_sem = max(self.config.semantic_weight, 0.0)
        w_kw = max(self.config.bm25_weight, 0.0)
        w_fz = max(self.config.fuzzy_weight, 0.0)
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
    # Candidate pooling utilities (for evaluation)
    # -----------------------------
    def candidate_pool(self, query: str, k_each: int = 20) -> List[dict]:
        """Return a deduplicated candidate set as union of top-k from semantic, bm25, and fuzzy."""
        sem_ids = self._top_k_ids(self._semantic_scores(query, self.rules), k_each)
        bm_ids = self._top_k_ids(self._bm25_scores(query, self.rules), k_each)
        fz_ids = self._top_k_ids(self._fuzzy_scores(query, self.rules), k_each)

        pool_ids = list(dict.fromkeys(sem_ids + bm_ids + fz_ids))  # preserve order, dedupe
        id_to_rule = {r["rule_id"]: r for r in self.rules}
        return [id_to_rule[rid] for rid in pool_ids if rid in id_to_rule]

    @staticmethod
    def _top_k_ids(score_dict: Dict[str, float], k: int) -> List[str]:
        return [rid for rid, _ in sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)[:k]]

    # -----------------------------
    # Rerank placeholder
    # -----------------------------
    def _rerank_results(self, query: str, results: List[dict], top_k: int) -> List[dict]:
        # Keep your existing reranker here if you have one; pass-through for now.
        return results[:top_k]
