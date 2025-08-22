"""
app.py — Application entrypoint

Purpose
-------
Bootstraps the Dash web application, initializes the SQLite database,
constructs retrieval components (embeddings + indices), and registers UI callbacks.

Key Responsibilities
--------------------
- Configure process-wide logging.
- Initialize Dash app (theme, fonts, layout).
- Initialize DatabaseManager (ingest CSV → SQLite if needed).
- Build the RuleRetriever (EmbeddingManager + indices).
- Register all Dash callbacks.
- Run the development server when executed directly.

Dependencies
------------
- dash, dash-bootstrap-components
- rag.embeddings.manager.EmbeddingManager
- rag.search.retriever.RuleRetriever
- rag.search.config.SearchConfig
- db.manager.DatabaseManager
- components.layout/search/generator (layout + callbacks)
- config constants (paths, flags, weights, etc.)
"""

from __future__ import annotations

import logging
from typing import Optional

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

from config import (
    SQLITE_DB_PATH,
    SQLITE_TABLE_NAME,
    CSV_DATA_PATH,
    APP_TITLE,
    DEBUG,
    HOST,
    PORT,
)

from rag.embeddings.manager import EmbeddingManager
from rag.search.retriever import RuleRetriever
from rag.search.config import SearchConfig
from components.layout import create_layout, register_layout_callbacks
from components.search import register_search_callbacks
from components.generator import register_generator_callbacks
from db.manager import DatabaseManager


# -------------------------------------------------------------------
# Logging setup (process-wide)
# -------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("app")

# Quiet down very chatty loggers in production; keep INFO in dev
logging.getLogger("werkzeug").setLevel(logging.INFO if DEBUG else logging.WARNING)


# -------------------------------------------------------------------
# App factory (keeps import side-effects minimal and helps testing)
# -------------------------------------------------------------------
def create_app() -> dash.Dash:
    """
    Create and configure the Dash application (without running it).

    Returns:
        dash.Dash: Configured Dash app instance.
    """
    logger.info("Initializing Dash application...")

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            # Important: must be a string, not a bare token
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
        ],
        suppress_callback_exceptions=True,  # allow callbacks to be registered after layout
        title=APP_TITLE,
    )

    # Expose WSGI server for gunicorn (`gunicorn app:server`)
    app.title = APP_TITLE
    app.server  # touch to ensure property exists
    logger.info("Dash app created (title=%s, debug=%s).", APP_TITLE, DEBUG)

    # Set base layout (static scaffolding)
    app.layout = create_layout()
    logger.info("Base layout attached.")

    return app


# -------------------------------------------------------------------
# Retriever initialization
# -------------------------------------------------------------------
def initialise_retriever(db_manager: DatabaseManager) -> RuleRetriever:
    """
    Construct a RuleRetriever with configured weights and embedding model.

    Args:
        db_manager: Initialized DatabaseManager instance.

    Returns:
        RuleRetriever: Ready-to-use retriever (indices will be built on init).
    """
    logger.info("Initializing RuleRetriever...")

    # Local import to avoid circulars and to keep config centralized here
    from config import (
        EMBEDDING_MODEL,
        SEMANTIC_WEIGHT,
        BM25_WEIGHT,
        FUZZY_WEIGHT,
        MIN_SIMILARITY,
        ENABLE_RERANKING,
    )

    # Build search configuration (weights are convex-combined at runtime)
    config = SearchConfig(
        semantic_weight=SEMANTIC_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        fuzzy_weight=FUZZY_WEIGHT,
        min_similarity=MIN_SIMILARITY,
        enable_reranking=ENABLE_RERANKING,
    )
    logger.debug(
        "SearchConfig(semantic=%.3f, bm25=%.3f, fuzzy=%.3f, min_sim=%.2f, rerank=%s)",
        config.semantic_weight,
        config.bm25_weight,
        config.fuzzy_weight,
        config.min_similarity,
        config.enable_reranking,
    )

    # Embedding manager (handles model loading, pooling, L2, caching)
    embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL)
    logger.info("EmbeddingManager created for model '%s'.", EMBEDDING_MODEL)

    # The retriever builds indices from rules loaded by the DB manager
    retriever = RuleRetriever(
        embedding_manager=embedding_manager,
        config=config,
        db_manager=db_manager,
    )
    logger.info("RuleRetriever initialized successfully (indices built).")
    return retriever


# -------------------------------------------------------------------
# Boot sequence
# -------------------------------------------------------------------
def boot() -> tuple[dash.Dash, RuleRetriever]:
    """
    Perform the full application boot sequence:
    - Create app
    - Initialize SQLite
    - Build retriever
    - Register callbacks

    Returns:
        (app, retriever): The Dash app and a ready-to-use RuleRetriever.
    """
    app = create_app()

    # Initialize database (idempotent: creates table/ingests CSV if needed)
    logger.info(
        "Initializing SQLite (db_path=%s, table=%s) with CSV='%s'...",
        SQLITE_DB_PATH,
        SQLITE_TABLE_NAME,
        CSV_DATA_PATH,
    )
    try:
        db_manager = DatabaseManager(db_path=SQLITE_DB_PATH, table_name=SQLITE_TABLE_NAME)
        db_manager.init_db(CSV_DATA_PATH)
        logger.info("SQLite initialization completed.")
    except Exception as exc:
        logger.critical("SQLite initialization failed: %s", exc, exc_info=True)
        # In production you might re-raise to fail fast:
        # raise
        # For dev, continue raising to avoid a half-initialized app
        raise

    # Build retriever (loads rules via db_manager internally)
    try:
        retriever = initialise_retriever(db_manager)
    except Exception as exc:
        logger.critical("Retriever initialization failed: %s", exc, exc_info=True)
        raise

    # Register callbacks (keep order stable)
    try:
        register_layout_callbacks(app)
        register_search_callbacks(app, retriever)
        register_generator_callbacks(app)
        logger.info("All callbacks registered.")
    except Exception as exc:
        logger.critical("Callback registration failed: %s", exc, exc_info=True)
        raise

    logger.info("Boot sequence completed. Application is ready.")
    return app, retriever


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
# Expose `server` at module top-level for gunicorn: `gunicorn app:server`
try:
    _app, _retriever = boot()
    server = _app.server
except Exception:
    # If boot fails, avoid exposing a broken `server` object.
    logger.critical("Application failed to boot. See logs above.", exc_info=True)
    raise


if __name__ == "__main__":
    logger.info("Starting development server at http://%s:%s (debug=%s)...", HOST, PORT, DEBUG)
    # Dash dev server — for production use gunicorn: `gunicorn app:server --workers 4`
    _app.run_server(debug=DEBUG, host=HOST, port=PORT)
