"""
RAG Retriever - Production-ready rule search and retrieval system
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from fuzzywuzzy import fuzz

from .embeddings import EmbeddingManager, EmbeddingIndex

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FUZZY = "fuzzy"


@dataclass
class SearchConfig:
    semantic_weight: float = 0.6
    bm25_weight: float = 0.3
    fuzzy_weight: float = 0.1
    min_similarity: float = 0.1
    max_results: int = 50
    enable_reranking: bool = True

class RuleRetriever:
    CSV_COLUMNS = {
        'rule_id': 'id',
        'rule_name': 'name',
        'rule_description': 'description',
        'bansta_code': 'bansta_code',
        'iso_code': 'iso_code',
        'business_division': 'division',
        'function': 'function',
        'tags': 'tags',
        'rule_body': 'rule_body',
        'english_description': 'description_en',
        'german_description': 'description_de',
        'llm_description_en': 'llm_description_en',
        'llm_description_de': 'llm_description_de',
        'embedding': 'embedding',
        'version_major': 'version_major',
        'version_minor': 'version_minor',
        'created_at': 'created_at',
        'updated_at': 'updated_at'
    }

    def __init__(self,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 config: Optional[SearchConfig] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.config = config or SearchConfig()
        self.rules_df = None
        self.embedding_index = None
        self.bm25_index = None
        self._rules_cache = {}

    def load_rules(self, filepath: str):
        path = Path(filepath)
        if path.suffix != ".csv":
            raise ValueError("Only CSV files are supported")

        df = pd.read_csv(filepath)
        self.rules_df = self._standardize_dataframe(df)
        self._build_indices()
        logger.info(f"Loaded {len(self.rules_df)} rules from {filepath}")

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        df = df.rename(columns={k: v for k, v in self.CSV_COLUMNS.items() if k in df.columns})

        for col in ["id", "name", "description", "category"]:
            if col not in df.columns:
                df[col] = "unknown"

        if "priority" not in df.columns:
            df["priority"] = "medium"

        if "id" not in df.columns or df["id"].isnull().any():
            df["id"] = [f"rule_{i}" for i in range(len(df))]

        df["name"] = df["name"].astype(str).fillna("")
        df["description"] = df["description"].astype(str).fillna("")

        return df

    def _build_indices(self):
        rules_list = self.rules_df.to_dict("records")
        self.embedding_index = EmbeddingIndex(self.embedding_manager)
        self.embedding_index.add_rules(rules_list)

        documents = [
            f"{row['name']} {row['description']}".lower().split()
            for _, row in self.rules_df.iterrows()
        ]
        self.bm25_index = BM25Okapi(documents)
        logger.info("Built search indices")

    def search_rules(
        self,
        query: Optional[str] = None,
        division: Optional[str] = None,
        priority: Optional[str] = None,
        function: Optional[str] = None,
        tags: Optional[Union[str, List[str]]] = None,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10
    ) -> pd.DataFrame:
        if self.rules_df is None:
            raise ValueError("No rules loaded. Call load_rules() first.")

        df = self.rules_df.copy()

        if division:
            df = df[df["division"].str.lower() == division.lower()]

        if priority:
            df = df[df["priority"].str.lower() == priority.lower()]

        if function:
            df = df[df["function"].str.lower() == function.lower()]

        if tags:
            if isinstance(tags, str):
                tags = [tags]
            df = df[df["tags"].apply(lambda tag_str: any(tag.lower() in tag_str.lower() for tag in tags if isinstance(tag_str, str)))]

        if not query:
            return df.head(top_k)

        if mode == SearchMode.SEMANTIC:
            scores = self._semantic_search(query, df)
        elif mode == SearchMode.KEYWORD:
            scores = self._keyword_search(query, df)
        elif mode == SearchMode.FUZZY:
            scores = self._fuzzy_search(query, df)
        else:
            scores = self._hybrid_search(query, df)

        df = df.copy()
        df["search_score"] = scores
        df = df[df["search_score"] >= self.config.min_similarity]

        if self.config.enable_reranking and len(df) > top_k:
            df = self._rerank_results(query, df, top_k)

        return df.sort_values("search_score", ascending=False).head(top_k)

    def _semantic_search(self, query: str, df: pd.DataFrame) -> np.ndarray:
        query_emb = self.embedding_manager.generate_embeddings(query)[0]
        embeddings = [
            self.embedding_manager.generate_embeddings(f"{row['name']} {row['description']}")[0]
            for _, row in df.iterrows()
        ]
        return cosine_similarity([query_emb], embeddings)[0] if embeddings else np.array([])

    def _keyword_search(self, query: str, df: pd.DataFrame) -> np.ndarray:
        documents = [
            f"{row['name']} {row['description']}".lower().split()
            for _, row in df.iterrows()
        ]
        bm25 = BM25Okapi(documents)
        scores = bm25.get_scores(query.lower().split())
        max_score = max(scores) if max(scores) > 0 else 1
        return np.array(scores) / max_score

    def _fuzzy_search(self, query: str, df: pd.DataFrame) -> np.ndarray:
        return np.array([
            max(
                fuzz.partial_ratio(query.lower(), f"{row['name']} {row['description']}".lower()),
                fuzz.token_sort_ratio(query.lower(), f"{row['name']} {row['description']}".lower())
            ) / 100.0
            for _, row in df.iterrows()
        ])

    def _hybrid_search(self, query: str, df: pd.DataFrame) -> np.ndarray:
        sem = self._semantic_search(query, df)
        kwd = self._keyword_search(query, df)
        fzy = self._fuzzy_search(query, df)
        return (
            self.config.semantic_weight * sem +
            self.config.bm25_weight * kwd +
            self.config.fuzzy_weight * fzy
        )

    def _rerank_results(self, query: str, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        rerank_df = df.head(top_k * 2).copy()
        rerank_scores = []

        for _, row in rerank_df.iterrows():
            score = row["search_score"]
            if query.lower() in row["name"].lower():
                score *= 1.2
            if row["priority"].lower() == "high":
                score *= 1.1
            if len(row["description"]) < 50:
                score *= 0.9
            rerank_scores.append(score)

        rerank_df["rerank_score"] = rerank_scores
        return rerank_df.sort_values("rerank_score", ascending=False)

    def get_rule_by_id(self, rule_id: str) -> Optional[Dict]:
        if rule_id in self._rules_cache:
            return self._rules_cache[rule_id]
        if self.rules_df is None:
            return None
        match = self.rules_df[self.rules_df["id"] == rule_id]
        if match.empty:
            return None
        rule_data = match.iloc[0].to_dict()
        self._rules_cache[rule_id] = rule_data
        return rule_data

    def get_similar_rules(
        self, rule_text: str, top_k: int = 5, exclude_ids: Optional[List[str]] = None
    ) -> List[Tuple[Dict, float]]:
        if self.embedding_index is None:
            raise ValueError("No embedding index available")
        results = self.embedding_index.search(rule_text, top_k * 2)
        if exclude_ids:
            results = [(rule, score) for rule, score in results if rule.get("id") not in exclude_ids]
        return results[:top_k]

    def update_rule_index(self, new_rules: Optional[List[Dict]] = None):
        if new_rules:
            new_df = pd.DataFrame(new_rules)
            self.rules_df = pd.concat([self.rules_df, new_df], ignore_index=True)
            self.rules_df.drop_duplicates(subset="id", keep="last", inplace=True)
        self._build_indices()
        self._rules_cache.clear()
        logger.info("Updated search indices")

    def get_search_statistics(self) -> Dict:
        if self.rules_df is None:
            return {}
        return {
            "total_rules": len(self.rules_df),
            "categories": self.rules_df["category"].value_counts().to_dict(),
            "priorities": self.rules_df["priority"].value_counts().to_dict(),
            "avg_description_length": self.rules_df["description"].str.len().mean(),
            "has_embedding_index": self.embedding_index is not None,
            "has_bm25_index": self.bm25_index is not None
        }
