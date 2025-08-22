"""
generator.py — Rule generator UI

Purpose
-------
Offer a drop-zone + chat interface to analyse existing rules and draft new ones
through a language-model backend.

Key Responsibilities
--------------------
- Build generator card: drop-zone, chips, chat.
- Manage dragged-in rules (add/remove).
- Register server and client callbacks for chat and DnD.

Dependencies
------------
- dash, dash_bootstrap_components, dash_iconify
- rag.generator.generate_response
- json
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import logging

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

# Local component factories
from .search import create_search_component
from .generator import create_generator_component

# Configuration (CHAT_ENABLED may be overridden by env or config.py)
try:
    from config import CHAT_ENABLED
except Exception:  # pragma: no cover
    CHAT_ENABLED = True  # safe default

# -----------------------------------------------------------------------------
# Module-wide logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Layout factory
# -----------------------------------------------------------------------------
def create_layout() -> dbc.Container:
    """
    Build and return the top-level Dash layout.

    Structure:
      • Header row with title & (optional) chat toggle button
      • Two columns:
          – Left : search component (always visible)
          – Right: rule-generator component (toggleable)
    """
    # ---------------------------- Header ----------------------------------
    # Chat toggle is visually present but hidden via style when CHAT_ENABLED is False.
    header = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        DashIconify(icon="mdi:brain", width=32, height=32, className="me-3"),
                        html.H1("RAG Rules Engine", className="mb-0 fw-bold"),
                        dbc.Button(
                            [
                                DashIconify(icon="mdi:chat-outline", width=22, height=22, className="me-2"),
                                html.Span(id="chat-toggle-label"),
                            ],
                            id="toggle-chat-collapse",
                            className="ms-4 px-4 py-2 rounded-pill shadow-sm fw-semibold",
                            color="light",
                            n_clicks=0,
                            style={
                                "fontSize": "1.1rem",
                                "transition": "background 0.2s, width 0.2s",
                                "background": "linear-gradient(90deg, #e0eafc 0%, #cfdef3 100%)",
                                "border": "1px solid #b6c6e3",
                                "color": "#2a3b5d",
                                "display": "block" if CHAT_ENABLED else "none",
                            },
                        ),
                        dbc.Tooltip(
                            id="chat-toggle-tooltip",
                            target="toggle-chat-collapse",
                            placement="bottom",
                        ),
                    ],
                    className="d-flex align-items-center py-3",
                )
            ),
        ],
        className="border-bottom mb-4",
    )

    # ---------------------------- Main split ------------------------------
    # Right column (chat) starts hidden; we keep a dcc.Store as the single source of truth.
    main = dbc.Row(
        [
            dbc.Col(
                create_search_component(),
                id="search-col",
                style={"width": "100%"},
                className="pe-3",
            ),
            dbc.Col(
                create_generator_component(),
                id="chat-col",
                style={"display": "none"},
                className="ps-3",
            ),
            dcc.Store(id="chat-col-visible", data=False),
        ],
        className="h-100",
    )

    # ---------------------------- Container -------------------------------
    logger.debug("Main layout created (CHAT_ENABLED=%s)", CHAT_ENABLED)
    return dbc.Container([header, main], fluid=True, className="main-container")


# -----------------------------------------------------------------------------
# Callback registrar
# -----------------------------------------------------------------------------
def register_layout_callbacks(app) -> None:
    """
    Register callbacks for toggling the chat column.

    Args:
        app: The Dash application instance.
    """

    def _on_state_to_styles(visible: bool) -> Tuple[Dict[str, Any], Dict[str, Any], bool, str, str]:
        """
        Convert the boolean visibility state to (chat style, search style, visible, label, tooltip).
        """
        if visible:
            # Split 50/50 when chat is open
            return (
                {"display": "block", "width": "50%"},
                {"width": "50%"},
                True,
                "Hide Chat",
                "Close Chat",
            )
        else:
            # Search takes full width when chat is hidden
            return (
                {"display": "none"},
                {"width": "100%"},
                False,
                "Show Chat",
                "Open Chat",
            )

    @app.callback(
        [
            Output("chat-col", "style"),
            Output("search-col", "style"),
            Output("chat-col-visible", "data"),
            Output("chat-toggle-label", "children"),
            Output("chat-toggle-tooltip", "children"),
        ],
        Input("toggle-chat-collapse", "n_clicks"),
        State("chat-col-visible", "data"),
        prevent_initial_call=False,
    )
    def toggle_chat_col(n_clicks: int, visible: bool):
        """
        Toggle the right-hand chat column between hidden and visible states.

        Behavior:
          - First render uses the existing 'visible' state from dcc.Store.
          - Each button click toggles the boolean.
        """
        # On first load (n_clicks is None/0), we keep current state.
        new_visible = not visible if (n_clicks or 0) > 0 else visible

        chat_style, search_style, store_val, label, tip = _on_state_to_styles(new_visible)
        logger.info(
            "Chat column %s (clicks=%s) -> visible=%s",
            "toggled" if (n_clicks or 0) > 0 else "initialized",
            n_clicks,
            store_val,
        )
        return chat_style, search_style, store_val, label, tip


__all__ = ["create_layout", "register_layout_callbacks"]
