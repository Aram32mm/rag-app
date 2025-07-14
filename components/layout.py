"""Main Application Layout
========================
Root *Dash* layout assembling the search & rule-generator panes into a
responsive two-column interface.

Public helper
-------------
* ``create_layout`` – returns the ready-to-mount :class:`dash.html.Div` tree.

The module purposefully contains **no callbacks**; each sub-component registers
its own when imported.
"""

from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

# Local component factories ---------------------------------------------------
from .search import create_search_component  
from .generator import create_generator_component  

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
        dbc.Col(
            html.Div(
                [
                    DashIconify(icon="mdi:brain", width=32, height=32, className="me-3"),
                    html.H1("RAG Rules Engine", className="mb-0 fw-bold"),
                ],
                className="d-flex align-items-center py-3",
            )
        ),
        className="border-bottom mb-4",
    )

    # ---------------------------- Main split ------------------------------
    main = dbc.Row(
        [
            dbc.Col(create_search_component(), width=6, className="pe-3"),
            dbc.Col(create_generator_component(), width=6, className="ps-3"),
        ],
        className="h-100",
    )

    # ---------------------------- Container -------------------------------
    return dbc.Container([header, main], fluid=True, className="main-container")

__all__ = ["create_layout"]