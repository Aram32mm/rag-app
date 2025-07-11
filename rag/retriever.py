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
import json
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from fuzzywuzzy import fuzz
import re

from embeddings import EmbeddingManager, EmbeddingIndex

logger = logging.getLogger(__name__)

class SearchMode(Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FUZZY = "fuzzy"

@dataclass
class SearchConfig:
    """Configuration for search behavior"""
    semantic_weight: float = 0.6
    bm25_weight: float = 0.3
    fuzzy_weight: float = 0.1
    min_similarity: float = 0.1
    max_results: int = 50
    enable_reranking: bool = True

class RuleRetriever:
    """Advanced rule retrieval system with multiple search strategies"""
    
    def __init__(self, 
                 embedding_manager: Optional[EmbeddingManager] = None,
                 config: Optional[SearchConfig] = None):
        """
        Initialize retriever
        
        Args:
            embedding_manager: Pre-configured embedding manager
            config: Search configuration
        """
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.config = config or SearchConfig()
        self.rules_df = None
        self.embedding_index = None
        self.bm25_index = None
        self._rules_cache = {}
        
    def load_rules(self, rules: Union[List[Dict], pd.DataFrame, str]):
        """
        Load rules from various sources
        
        Args:
            rules: Rules as list, DataFrame, or file path
        """
        if isinstance(rules, str):
            rules = self._load_rules_from_file(rules)
        
        if isinstance(rules, list):
            self.rules_df = pd.DataFrame(rules)
        else:
            self.rules_df = rules.copy()
        
        # Standardize columns
        self._standardize_dataframe()
        
        # Build indices
        self._build_indices()
        
        logger.info(f"Loaded {len(self.rules_df)} rules")
    
    def _load_rules_from_file(self, filepath: str) -> pd.DataFrame:
        """Load rules from file"""
        path = Path(filepath)
        
        if path.suffix == '.csv':
            return pd.read_csv(filepath)
        elif path.suffix == '.json':
            with open(filepath) as f:
                return pd.DataFrame(json.load(f))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _standardize_dataframe(self):
        """Standardize DataFrame column names and content"""
        # Map common column variations
        column_mapping = {
            'rule_name': 'name',
            'rule_description': 'description',
            'rule_body': 'description',
            'generated_description': 'description',
            'rule_category': 'category',
            'rule_priority': 'priority'
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in self.rules_df.columns:
                self.rules_df = self.rules_df.rename(columns={old_name: new_name})
        
        # Ensure required columns exist
        required_columns = ['name', 'description', 'category', 'priority']
        for col in required_columns:
            if col not in self.rules_df.columns:
                self.rules_df[col] = 'unknown'
        
        # Add ID if missing
        if 'id' not in self.rules_df.columns:
            self.rules_df['id'] = [f"rule_{i}" for i in range(len(self.rules_df))]
        
        # Clean text data
        text_columns = ['name', 'description']
        for col in text_columns:
            self.rules_df[col] = self.rules_df[col].astype(str).fillna('')
    
    def _build_indices(self):
        """Build search indices"""
        # Build embedding index
        self.embedding_index = EmbeddingIndex(self.embedding_manager)
        rules_list = self.rules_df.to_dict('records')
        self.embedding_index.add_rules(rules_list)
        
        # Build BM25 index
        documents = []
        for _, row in self.rules_df.iterrows():
            # Combine name and description for full-text search
            doc_text = f"{row['name']} {row['description']}"
            documents.append(doc_text.lower().split())
        
        self.bm25_index = BM25Okapi(documents)
        
        logger.info("Built search indices")
    
    def search_rules(self,
                    query: Optional[str] = None,
                    category: Optional[str] = None,
                    priority: Optional[str] = None,
                    mode: SearchMode = SearchMode.HYBRID,
                    top_k: int = 10) -> pd.DataFrame:
        """
        Advanced rule search with multiple strategies
        
        Args:
            query: Search query text
            category: Filter by category
            priority: Filter by priority
            mode: Search mode
            top_k: Number of results
            
        Returns:
            DataFrame with search results and scores
        """
        if self.rules_df is None:
            raise ValueError("No rules loaded. Call load_rules() first.")
        
        # Start with all rules
        results_df = self.rules_df.copy()
        
        # Apply filters first
        if category:
            results_df = results_df[results_df['category'].str.lower() == category.lower()]
        
        if priority:
            results_df = results_df[results_df['priority'].str.lower() == priority.lower()]
        
        # If no query, return filtered results
        if not query:
            return results_df.head(top_k)
        
        # Apply search based on mode
        if mode == SearchMode.SEMANTIC:
            scores = self._semantic_search(query, results_df)
        elif mode == SearchMode.KEYWORD:
            scores = self._keyword_search(query, results_df)
        elif mode == SearchMode.FUZZY:
            scores = self._fuzzy_search(query, results_df)
        else:  # HYBRID
            scores = self._hybrid_search(query, results_df)
        
        # Add scores and sort
        results_df['search_score'] = scores
        results_df = results_df[results_df['search_score'] >= self.config.min_similarity]
        results_df = results_df.sort_values('search_score', ascending=False)
        
        # Apply reranking if enabled
        if self.config.enable_reranking and len(results_df) > top_k:
            results_df = self._rerank_results(query, results_df, top_k)
        
        return results_df.head(top_k)
    
    def _semantic_search(self, query: str, df: pd.DataFrame) -> np.ndarray:
        """Semantic search using embeddings"""
        if len(df) == 0:
            return np.array([])
        
        query_embedding = self.embedding_manager.generate_embeddings(query)[0]
        
        # Get embeddings for filtered rules
        rule_embeddings = []
        for idx in df.index:
            rule_data = df.loc[idx].to_dict()
            rule_text = f"{rule_data['name']} {rule_data['description']}"
            embedding = self.embedding_manager.generate_embeddings(rule_text)[0]
            rule_embeddings.append(embedding)
        
        if not rule_embeddings:
            return np.array([])
        
        rule_embeddings = np.array(rule_embeddings)
        similarities = cosine_similarity([query_embedding], rule_embeddings)[0]
        
        return similarities
    
    def _keyword_search(self, query: str, df: pd.DataFrame) -> np.ndarray:
        """BM25-based keyword search"""
        if len(df) == 0:
            return np.array([])
        
        # Create BM25 index for filtered results
        documents = []
        for _, row in df.iterrows():
            doc_text = f"{row['name']} {row['description']}"
            documents.append(doc_text.lower().split())
        
        if not documents:
            return np.array([])
        
        bm25 = BM25Okapi(documents)
        scores = bm25.get_scores(query.lower().split())
        
        # Normalize scores to [0, 1]
        if len(scores) > 0:
            max_score = max(scores) if max(scores) > 0 else 1
            scores = scores / max_score
        
        return scores
    
    def _fuzzy_search(self, query: str, df: pd.DataFrame) -> np.ndarray:
        """Fuzzy string matching search"""
        if len(df) == 0:
            return np.array([])
        
        scores = []
        for _, row in df.iterrows():
            # Combine name and description for fuzzy matching
            combined_text = f"{row['name']} {row['description']}"
            
            # Use multiple fuzzy matching strategies
            partial_ratio = fuzz.partial_ratio(query.lower(), combined_text.lower())
            token_ratio = fuzz.token_sort_ratio(query.lower(), combined_text.lower())
            
            # Take the maximum of different fuzzy scores
            fuzzy_score = max(partial_ratio, token_ratio) / 100.0
            scores.append(fuzzy_score)
        
        return np.array(scores)
    
    def _hybrid_search(self, query: str, df: pd.DataFrame) -> np.ndarray:
        """Hybrid search combining multiple strategies"""
        if len(df) == 0:
            return np.array([])
        
        # Get scores from different methods
        semantic_scores = self._semantic_search(query, df)
        keyword_scores = self._keyword_search(query, df)
        fuzzy_scores = self._fuzzy_search(query, df)
        
        # Combine scores with weights
        combined_scores = (
            semantic_scores * self.config.semantic_weight +
            keyword_scores * self.config.bm25_weight +
            fuzzy_scores * self.config.fuzzy_weight
        )
        
        return combined_scores
    
    def _rerank_results(self, query: str, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """Rerank top results using additional signals"""
        # Take top 2*k results for reranking
        rerank_df = df.head(top_k * 2)
        
        # Add additional ranking signals
        rerank_scores = []
        
        for _, row in rerank_df.iterrows():
            score = row['search_score']
            
            # Boost exact matches in name
            if query.lower() in row['name'].lower():
                score *= 1.2
            
            # Boost high priority rules
            if row['priority'].lower() == 'high':
                score *= 1.1
            
            # Penalize very short descriptions
            if len(row['description']) < 50:
                score *= 0.9
            
            rerank_scores.append(score)
        
        rerank_df['rerank_score'] = rerank_scores
        return rerank_df.sort_values('rerank_score', ascending=False)
    
    def get_rule_by_id(self, rule_id: str) -> Optional[Dict]:
        """
        Retrieve specific rule by ID
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            Rule data or None
        """
        if rule_id in self._rules_cache:
            return self._rules_cache[rule_id]
        
        if self.rules_df is None:
            return None
        
        matches = self.rules_df[self.rules_df['id'] == rule_id]
        if len(matches) == 0:
            return None
        
        rule_data = matches.iloc[0].to_dict()
        self._rules_cache[rule_id] = rule_data
        return rule_data
    
    def get_similar_rules(self, 
                         rule_text: str, 
                         top_k: int = 5,
                         exclude_ids: Optional[List[str]] = None) -> List[Tuple[Dict, float]]:
        """
        Find rules similar to given text
        
        Args:
            rule_text: Reference text
            top_k: Number of results
            exclude_ids: Rule IDs to exclude
            
        Returns:
            List of (rule, score) tuples
        """
        if self.embedding_index is None:
            raise ValueError("No embedding index available")
        
        results = self.embedding_index.search(rule_text, top_k * 2)
        
        # Filter out excluded IDs
        if exclude_ids:
            results = [(rule, score) for rule, score in results 
                      if rule.get('id') not in exclude_ids]
        
        return results[:top_k]
    
    def update_rule_index(self, new_rules: Optional[List[Dict]] = None):
        """
        Update search indices with new rules
        
        Args:
            new_rules: New rules to add (optional)
        """
        if new_rules:
            new_df = pd.DataFrame(new_rules)
            self.rules_df = pd.concat([self.rules_df, new_df], ignore_index=True)
        
        # Rebuild indices
        self._build_indices()
        
        # Clear cache
        self._rules_cache.clear()
        
        logger.info("Updated search indices")
    
    def get_search_statistics(self) -> Dict:
        """Get statistics about the search index"""
        if self.rules_df is None:
            return {}
        
        stats = {
            'total_rules': len(self.rules_df),
            'categories': self.rules_df['category'].value_counts().to_dict(),
            'priorities': self.rules_df['priority'].value_counts().to_dict(),
            'avg_description_length': self.rules_df['description'].str.len().mean(),
            'has_embedding_index': self.embedding_index is not None,
            'has_bm25_index': self.bm25_index is not None
        }
        
        return stats

# Convenience functions for backward compatibility
def search_rules(query: Optional[str] = None,
                category: Optional[str] = None,
                priority: Optional[str] = None,
                retriever: Optional[RuleRetriever] = None) -> List[Dict]:
    """
    Legacy search function for backward compatibility
    """
    if retriever is None:
        # Use mock data if no retriever provided
        mock_rules = [
            {
                "id": "rule_1",
                "name": "Customer Age Validation",
                "description": "Validate that customer age is between 18 and 120 years for account creation",
                "category": "business",
                "priority": "high"
            },
            {
                "id": "rule_2", 
                "name": "Password Complexity",
                "description": "Ensure passwords contain at least 8 characters with uppercase, lowercase, numbers and symbols",
                "category": "technical",
                "priority": "high"
            },
            {
                "id": "rule_3",
                "name": "Transaction Limit Check",
                "description": "Daily transaction limit should not exceed $10,000 for standard accounts",
                "category": "business", 
                "priority": "medium"
            },
            {
                "id": "rule_4",
                "name": "Data Retention Policy",
                "description": "Customer data must be retained for 7 years as per compliance requirements",
                "category": "compliance",
                "priority": "high"
            },
            {
                "id": "rule_5",
                "name": "Email Format Validation",
                "description": "Validate email format using RFC 5322 standard",
                "category": "technical",
                "priority": "low"
            }
        ]
        
        results = mock_rules
        
        if category:
            results = [r for r in results if r["category"] == category]
        
        if priority:
            results = [r for r in results if r["priority"] == priority]
            
        if query:
            query_lower = query.lower()
            results = [r for r in results if 
                      query_lower in r["name"].lower() or 
                      query_lower in r["description"].lower()]
        
        return results
    
    # Use actual retriever
    results_df = retriever.search_rules(query, category, priority)
    return results_df.to_dict('records')

def get_rule_by_id(rule_id: str, retriever: Optional[RuleRetriever] = None) -> Optional[Dict]:
    """Legacy function for backward compatibility"""
    if retriever is None:
        return None
    return retriever.get_rule_by_id(rule_id)

def get_similar_rules(rule_text: str, 
                     top_k: int = 5,
                     retriever: Optional[RuleRetriever] = None) -> List[Tuple[Dict, float]]:
    """Legacy function for backward compatibility"""
    if retriever is None:
        return []
    return retriever.get_similar_rules(rule_text, top_k)

def update_rule_index(new_rules: Optional[List[Dict]] = None,
                     retriever: Optional[RuleRetriever] = None):
    """Legacy function for backward compatibility"""
    if retriever is not None:
        retriever.update_rule_index(new_rules)
