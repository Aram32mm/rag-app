"""
layout.py — Application layout

Purpose
-------
Assemble the top-level layout with header and a responsive split between
search and rule generator panes.

Key Responsibilities
--------------------
- Expose `create_layout()` top-level container.
- Provide a chat-pane toggle callback.

Dependencies
------------
- dash, dash_bootstrap_components, dash_iconify
- components.search, components.generator
- config (CHAT_ENABLED)
"""

from __future__ import annotations

from dash import html, Input, Output, State
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
from dash import dcc

# Local component factories ---------------------------------------------------
from .search import create_search_component
from .generator import create_generator_component
from config import CHAT_ENABLED


def create_layout() -> dbc.Container:
    """Return the top-level Dash *layout*.

    The layout is organised as:

    * Header row with title & icon
    * Two equal-width columns beneath:
        – **Left**: search component
        – **Right**: rule-generator component
    """
    # ---------------------------- Header ----------------------------------
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
                ),
            ),
        ],
        className="border-bottom mb-4",
    )

    # ---------------------------- Main split ------------------------------
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
    return dbc.Container([header, main], fluid=True, className="main-container")


def register_layout_callbacks(app):
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
    )
    def toggle_chat_col(n_clicks, visible):
        if n_clicks:
            visible = not visible
        if visible:
            return {"display": "block", "width": "50%"}, {"width": "50%"}, True, "Hide Chat", "Close Chat"
        else:
            return {"display": "none"}, {"width": "100%"}, False, "", "Open Chat"


__all__ = ["create_layout"]
