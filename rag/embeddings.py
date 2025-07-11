"""
RAG Embeddings - Production-ready text embedding and similarity search
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Production-ready embedding manager with caching and batch processing"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize embedding manager
        
        Args:
            model_name: HuggingFace model name or local path
            device: Target device (auto-detected if None)
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.device = self._get_device() if device is None else device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = None
        self.tokenizer = None
        self._embedding_cache = {}
        
    def _get_device(self) -> str:
        """Auto-detect optimal device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_model(self):
        """Load embedding model with error handling"""
        try:
            if "sentence-transformers" in self.model_name:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            
            logger.info(f"Loaded model {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_embeddings(self, 
                          texts: Union[str, List[str]], 
                          batch_size: int = 32,
                          use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings with batching and caching
        
        Args:
            texts: Single text or list of texts
            batch_size: Processing batch size
            use_cache: Enable embedding caching
            
        Returns:
            Embeddings array
        """
        if self.model is None:
            self.load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._compute_batch_embeddings(batch, use_cache)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _compute_batch_embeddings(self, batch: List[str], use_cache: bool) -> List[np.ndarray]:
        """Compute embeddings for a batch with caching"""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(batch):
            text_hash = hash(text)
            if use_cache and text_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[text_hash])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if uncached_texts:
            if isinstance(self.model, SentenceTransformer):
                new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            else:
                new_embeddings = self._compute_transformer_embeddings(uncached_texts)
            
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings[idx] = embedding
                if use_cache:
                    self._embedding_cache[hash(batch[idx])] = embedding
        
        return embeddings
    
    def _compute_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings using standard transformers"""
        inputs = self.tokenizer(texts, return_tensors='pt', 
                               truncation=True, padding=True, 
                               max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()

def embed_rule(rule: Dict) -> np.ndarray:
    """
    Generate embedding for a single rule
    
    Args:
        rule: Rule dictionary with name and description
        
    Returns:
        Rule embedding vector
    """
    embedding_manager = EmbeddingManager()
    
    # Combine rule components for richer representation
    rule_text = f"{rule.get('name', '')} {rule.get('description', '')}"
    if rule.get('category'):
        rule_text += f" category:{rule['category']}"
    
    return embedding_manager.generate_embeddings(rule_text)[0]

def compute_similarity(embedding1: np.ndarray, 
                      embedding2: np.ndarray,
                      metric: str = "cosine") -> float:
    """
    Compute similarity between embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
    Returns:
        Similarity score
    """
    if metric == "cosine":
        return cosine_similarity([embedding1], [embedding2])[0][0]
    elif metric == "euclidean":
        return 1 / (1 + np.linalg.norm(embedding1 - embedding2))
    elif metric == "dot":
        return np.dot(embedding1, embedding2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def find_similar_embeddings(query_embedding: np.ndarray,
                           rule_embeddings: List[Dict],
                           top_k: int = 5,
                           threshold: float = 0.0) -> List[Tuple[Dict, float]]:
    """
    Find most similar embeddings with advanced filtering
    
    Args:
        query_embedding: Query embedding vector
        rule_embeddings: List of dicts with 'embedding' and metadata
        top_k: Number of results to return
        threshold: Minimum similarity threshold
        
    Returns:
        List of (rule, similarity_score) tuples
    """
    similarities = []
    
    for rule_data in rule_embeddings:
        similarity = compute_similarity(query_embedding, rule_data['embedding'])
        if similarity >= threshold:
            similarities.append((rule_data, similarity))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

class EmbeddingIndex:
    """Vector index for efficient similarity search"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.embeddings = []
        self.metadata = []
        self.index_built = False
    
    def add_rules(self, rules: List[Dict]):
        """Add rules to the index"""
        for rule in rules:
            embedding = embed_rule(rule)
            self.embeddings.append(embedding)
            self.metadata.append(rule)
        
        self.embeddings = np.array(self.embeddings)
        self.index_built = True
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search the index"""
        if not self.index_built:
            raise ValueError("Index not built. Add rules first.")
        
        query_embedding = self.embedding_manager.generate_embeddings(query)[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.metadata[idx], similarities[idx]))
        
        return results
    
    def save(self, filepath: str):
        """Save index to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'metadata': self.metadata,
                'index_built': self.index_built
            }, f)
    
    def load(self, filepath: str):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            self.index_built = data['index_built']

def update_embedding_index(rules: List[Dict], 
                          index_path: Optional[str] = None) -> EmbeddingIndex:
    """
    Update or create embedding index
    
    Args:
        rules: List of rules to index
        index_path: Path to save/load index
        
    Returns:
        Updated embedding index
    """
    embedding_manager = EmbeddingManager()
    index = EmbeddingIndex(embedding_manager)
    
    # Load existing index if available
    if index_path and Path(index_path).exists():
        try:
            index.load(index_path)
            logger.info(f"Loaded existing index from {index_path}")
        except Exception as e:
            logger.warning(f"Failed to load index: {e}. Creating new index.")
    
    # Add new rules
    index.add_rules(rules)
    
    # Save updated index
    if index_path:
        index.save(index_path)
        logger.info(f"Saved index to {index_path}")
    
    return index

def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingManager:
    """
    Load embedding model with configuration
    
    Args:
        model_name: Model identifier
        
    Returns:
        Configured embedding manager
    """
    manager = EmbeddingManager(model_name)
    manager.load_model()
    return manager