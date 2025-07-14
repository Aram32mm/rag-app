"""
Refactored Dash rule-search component (single back-end entry point).

Key points
-----------
✓ Auto-run on debounce / filter changes (no Search button).
✓ “Exact match” switch → keyword-only; otherwise hybrid blend.
✓ Filters in an Offcanvas sidebar, multi-select.
✓ Drag-and-drop cards retained, with term highlighting & similarity badge.
✓ ONE back-end call: `_retriever.search_rules` (no silent fallback).
"""

from __future__ import annotations

import json
import re
import time
from typing import List

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update
from dash_iconify import DashIconify
from dash.dependencies import ALL

from rag.retriever import SearchMode

# ---------------------------------------------------------------------------
# UI constants                                                               
# ---------------------------------------------------------------------------

CATEGORY_OPTIONS = [
    {"label": l, "value": v}
    for l, v in [
        ("Business", "business"),
        ("Technical", "technical"),
        ("Compliance", "compliance"),
        ("Security", "security"),
        ("Validation", "validation"),
    ]
]
PRIORITY_OPTIONS = [
    {"label": l, "value": v}
    for l, v in [("High", "high"), ("Medium", "medium"), ("Low", "low")]
]
PRIORITY_COLOURS = {"high": "danger", "medium": "warning", "low": "success"}
CATEGORY_COLOURS = {
    "business": "primary",
    "technical": "info",
    "compliance": "secondary",
    "security": "danger",
    "validation": "warning",
}

# ---------------------------------------------------------------------------
# Layout helpers                                                             
# ---------------------------------------------------------------------------

def create_search_component() -> dbc.Card:
    """Return a ready-to-use search section (Card)."""

    # Off-canvas filters
    offcanvas = dbc.Offcanvas(
        id="filters-offcanvas",
        title="Rule facets",
        placement="start",
        backdrop=True,
        children=[
            html.H6("Category", className="mt-3"),
            dbc.Checklist(id="category-filter", options=CATEGORY_OPTIONS, value=[], inline=False, className="mb-3"),
            html.H6("Priority"),
            dbc.Checklist(id="priority-filter", options=PRIORITY_OPTIONS, value=[], inline=False),
            html.H6("Function", className="mt-3"),
            dbc.Select(id="function-filter", options=[], value=None, placeholder="Select function", className="mb-3"),
            html.H6("Tags"),
            dbc.Select(id="tags-filter", options=[], value=None, placeholder="Select tag", className="mb-3"),

        ],
    )

    # Header
    header = dbc.CardHeader(
        html.Div(
            [
                html.H4([DashIconify(icon="mdi:magnify", className="me-2"), "Search Rules"], className="mb-0 fw-semibold"),
                dbc.Button(DashIconify(icon="mdi:tune"), id="open-filters", color="outline-primary", size="sm", className="ms-auto", title="Show filters"),
            ],
            className="d-flex align-items-center",
        ),
        className="bg-white shadow-sm",
    )

    # Search bar + switch
    search_row = html.Div(
        [
            dbc.Input(
                id="search-input",
                placeholder="Search rules …",
                debounce=True,
                autoComplete="off",
                className="search-input-simple me-2 flex-grow-1",
            ),
            dbc.Button(
                DashIconify(icon="mdi:close"),
                id="clear-search",
                color="outline-secondary",
                size="sm",
                title="Clear search",
                className="clear-btn-simple me-md-3 mb-2 mb-md-0",
            ),
            dbc.Switch(
                id="exact-switch",
                label="Exact match",
                className="ms-md-2",
                value=False,
            ),
        ],
        className="d-flex flex-column flex-md-row align-items-start align-items-md-center mb-3",
    )


    # Results container
    results_container = dcc.Loading(
        id="search-loading",
        type="circle",
        children=[html.Div(id="search-stats", className="mb-3"), html.Div(id="search-results")],
    )

    # Modal for full rule details
    modal = dbc.Modal(
        [
            dbc.ModalHeader("Rule Details", close_button=True),
            dbc.ModalBody(id="rule-modal-body"),
        ],
        id="rule-detail-modal",
        size="lg",
        is_open=False,
        scrollable=True,
    )

    return html.Div([  # Wrap Card + Modal together
        dbc.Card([header, dbc.CardBody([search_row, results_container]), offcanvas], className="h-100 shadow-sm"),
        modal
    ])


# ---------------------------------------------------------------------------
# Helper: highlight query terms                                               
# ---------------------------------------------------------------------------

def _highlight(text: str, terms: List[str]) -> str:  # noqa: D401
    if not terms:
        return text
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.I)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

# ---------------------------------------------------------------------------
# Result card builder                                                        
# ---------------------------------------------------------------------------

def make_rule_card(rule: dict, score: float | None, q_terms: List[str]) -> html.Div:
    priority_colour = PRIORITY_COLOURS.get(rule.get("priority", "low"), "secondary")
    category_colour = CATEGORY_COLOURS.get(rule.get("category", "business"), "primary")

    # Parse and render tags properly
    tag_str = rule.get("tags", "")
    try:
        # Handles both raw comma-separated string or Python list-string
        tags = eval(tag_str) if isinstance(tag_str, str) and tag_str.startswith("[") else tag_str.split(",")
    except Exception:
        tags = []

    card = dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H6(rule.get("name", "Untitled Rule"), className="fw-bold mb-2 flex-grow-1"),
                        dbc.Badge(
                            [DashIconify(icon="mdi:target", className="me-1"), f"{score:.1%}"],
                            color="success",
                            className="ms-auto" if score is not None else "d-none"
                        ),
                    ],
                    className="d-flex align-items-start",
                ),

                dcc.Markdown(
                    _highlight(rule.get("description", "No description"), q_terms),
                    dangerously_allow_html=True,
                    className="text-muted mb-3"
                ),

                html.Div(
                    [
                        dbc.Badge(
                            [DashIconify(icon="mdi:domain", className="me-1"), rule.get("division", "Unknown").title()],
                            color=category_colour,
                            className="me-2"
                        ) if rule.get("category") else None,

                        dbc.Badge(
                            [DashIconify(icon="mdi:priority-high", className="me-1"), rule.get("priority", "Medium").title()],
                            color=priority_colour
                        ) if rule.get("priority") else None,

                        dbc.Badge(
                            [DashIconify(icon="mdi:function-variant", className="me-1"), rule["function"].title()],
                            color="dark",
                            className="ms-2"
                        ) if rule.get("function") else None,
                    ],
                    className="mb-2 d-flex flex-wrap"
                ),

                html.Div(
                    [
                        dbc.Badge(
                            tag.strip(),
                            color="secondary",
                            className="me-1 mb-1"
                        )
                        for tag in tags if tag.strip()
                    ],
                    className="d-flex flex-wrap"
                ),

                # Still show a subtle drag icon (optional)
                DashIconify(icon="mdi:drag", className="drag-handle position-absolute top-0 end-0 p-2"),
            ]
        ),
        className="rule-card shadow-sm"
    )

    return html.Div(
        card,
        id={"type": "rule-card", "index": rule["id"]},
        className="mb-2 rule-card rule-card-clickable",
        n_clicks=0,
        draggable="true",
        style={"cursor": "grab"},
        **{"data-rule": json.dumps(rule)}
    )


# ---------------------------------------------------------------------------
# Callback registration                                                       
# ---------------------------------------------------------------------------

def register_search_callbacks(app, retriever):  # noqa: D401
    """Wire Dash callbacks – call from the parent layout factory."""

    # Toggle filter sidebar
    @app.callback(Output("filters-offcanvas", "is_open"), Input("open-filters", "n_clicks"), State("filters-offcanvas", "is_open"))
    def _toggle_filters(n, is_open):  # noqa: D401
        return not is_open if n else is_open

    # Clear search input
    @app.callback(Output("search-input", "value", allow_duplicate=True), Input("clear-search", "n_clicks"), prevent_initial_call=True)
    def _clear_input(n):  # noqa: D401
        return "" if n else ""

    # Main search (auto-run)
    @app.callback(
        [Output("search-results", "children"), Output("search-stats", "children")],
        [Input("search-input", "value"),
         Input("category-filter", "value"),
         Input("priority-filter", "value"),
         Input("function-filter", "value"),
         Input("tags-filter", "value"),
         Input("exact-switch", "value")]
    )
    def _search(query, divisions, prios, function, tag, exact):
        if retriever is None:
            return [html.Div("Search backend not initialised", className="text-danger")], None

        mode = SearchMode.KEYWORD if exact else SearchMode.HYBRID

        if not (query or divisions or prios or function or tag):
            empty = html.Div([
                DashIconify(icon="mdi:file-search-outline", width=48, className="text-muted mb-3"),
                html.P("Start typing to search rules …", className="text-muted"),
            ], className="text-center py-5")
            return empty, ""

        start = time.time()
        try:
            df = retriever.search_rules(
                query=query,
                division=divisions[0] if divisions else None,
                priority=prios[0] if prios else None,
                function=function or None,
                tags=tag or None,
                mode=mode,
                top_k=20
            )
            rows = df.to_dict("records") if not df.empty else []
            elapsed = time.time() - start

            if not rows:
                no_res = html.Div([
                    DashIconify(icon="mdi:file-remove-outline", width=48, className="text-muted mb-3"),
                    html.P("No rules matched your criteria", className="text-muted"),
                ], className="text-center py-5")
                stats = dbc.Alert([
                    DashIconify(icon="mdi:information", className="me-2"),
                    f"0 matches · {elapsed:.3f}s"
                ], color="warning", className="mb-0")
                return no_res, stats

            q_terms = query.split() if query else []
            cards = [make_rule_card(r, r.get("search_score"), q_terms) for r in rows]
            stats = dbc.Alert([
                DashIconify(icon="mdi:check-circle", className="me-2"),
                f"{len(cards)} rule(s) · {elapsed:.3f}s"
            ], color="success", className="mb-3")
            return cards, stats

        except Exception as exc:
            err_view = html.Div([
                DashIconify(icon="mdi:alert-circle", width=48, className="text-danger mb-3"),
                html.P(f"Search error: {exc}", className="text-danger"),
            ], className="text-center py-5")
            alert = dbc.Alert([
                DashIconify(icon="mdi:alert", className="me-2"),
                f"Search failed: {exc}"
            ], color="danger", className="mb-0")
            return err_view, alert
    
    from dash.dependencies import ALL  # at top

    @app.callback(
        Output("rule-detail-modal", "is_open"),
        Output("rule-modal-body", "children"),
        Input({"type": "rule-card", "index": ALL}, "n_clicks"),
        State({"type": "rule-card", "index": ALL}, "data-rule"),
        State("rule-detail-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _display_rule_details(clicks, rules_data, is_open):
        if not any(clicks):
            return is_open, no_update

        # Get clicked rule
        clicked_idx = next(i for i, c in enumerate(clicks) if c)
        rule = json.loads(rules_data[clicked_idx])

        EXCLUDED = {"embedding", "created_at", "updated_at"}

        # Create modal body with all fields
        rows = [
            html.Div([
                html.H6(k.replace("_", " ").title(), className="fw-bold"),
                html.Div(str(v), className="mb-3 text-muted"),
            ]) for k, v in rule.items() if k not in EXCLUDED and v not in [None, "", []]
        ]

        return True, rows


__all__ = ["register_search_callbacks", "create_search_component"]
