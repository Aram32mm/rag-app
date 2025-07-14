import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

from typing import Optional
from rag.embeddings import EmbeddingManager
from rag.retriever import RuleRetriever, SearchConfig

from components.layout import create_layout
from components.search import register_search_callbacks
from components.generator import register_generator_callbacks

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True
)

app.title = "RAG Rules Engine"
_retriever: Optional[RuleRetriever] = None
server = app.server


def initialise_retriever(rules_path: str | None = None) -> None:  # noqa: D401
    """Create and cache a global RuleRetriever."""
    global _retriever
    if _retriever is not None:
        return  # Already done

    config = SearchConfig(
        semantic_weight=0.7,
        bm25_weight=0.2,
        fuzzy_weight=0.1,
        min_similarity=0.15,
        enable_reranking=True,
    )
    embedding_manager = EmbeddingManager(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="./embedding_cache",
    )
    _retriever = RuleRetriever(embedding_manager, config)
    if rules_path:
        _retriever.load_rules(rules_path)

# Set layout
app.layout = create_layout()

# Initialize retriever with rules data
initialise_retriever("dummy.csv") 

# Register callbacks
register_search_callbacks(app, _retriever)
register_generator_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
