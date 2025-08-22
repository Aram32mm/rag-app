"""
search.py — Search UI component

Purpose
-------
Provide the rule search interface with filters, exact/hybrid switch, results list,
and a detail modal; wire it to the retriever.

Key Responsibilities
--------------------
- Build search card, offcanvas filters, and results container.
- Normalize/prepare scores for display.
- Register callbacks for filtering, searching, and modal details.

Dependencies
------------
- dash, dash_bootstrap_components, dash_iconify
- rag.search.config.SearchMode
- logging
"""


from __future__ import annotations

import json
import time
import logging
from typing import List, Optional

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update
from dash_iconify import DashIconify
from dash.dependencies import ALL

from rag.search.config import SearchMode

# ---------------------------------------------------------------------------
# Module-wide logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def create_search_component() -> dbc.Card:
    """
    Build the full search UI component (wrapped in a Card).
    
    Includes:
    - Offcanvas sidebar with filters
    - Header with search title + filter button
    - Search input + exact-match toggle
    - Results container
    - Rule detail modal
    """

    # --- Offcanvas filters (multi-select dropdowns for facets) ---
    offcanvas = dbc.Offcanvas(
        id="filters-offcanvas",
        title="Rule facets",
        placement="start",
        backdrop=True,
        children=[
            html.H6("Rule Type", className="mt-3"),
            dcc.Dropdown(
                id="rule-type-filter",
                options=[], value=None, multi=True, searchable=True,
                placeholder="Select rule type(s)", className="mb-3"
            ),

            html.H6("Country"),
            dcc.Dropdown(
                id="country-filter",
                options=[], value=None, multi=True, searchable=True,
                placeholder="Select country(s)", className="mb-3"
            ),

            html.H6("Business Team"),
            dcc.Dropdown(
                id="business-team-filter",
                options=[], value=None, multi=True, searchable=True,
                placeholder="Select business team(s)", className="mb-3"
            ),

            html.H6("Party Agent"),
            dcc.Dropdown(
                id="party-agent-filter",
                options=[], value=None, multi=True, searchable=True,
                placeholder="Select party agent(s)", className="mb-3"
            ),

            html.Hr(),
            dbc.Button(
                "Clear Filters", id="clear-filters-btn",
                color="outline-danger", className="mb-3 w-100"
            ),
        ],
    )

    # --- Header section with title + filters button ---
    header = dbc.CardHeader(
        html.Div(
            [
                html.H4(
                    [DashIconify(icon="mdi:magnify", className="me-2"), "Search Rules"],
                    className="mb-0 fw-semibold"
                ),
                dbc.Button(
                    [DashIconify(icon="mdi:tune", className="me-1"), "Filters"],
                    id="open-filters",
                    color="primary", size="md",
                    className="filter-btn ms-auto",
                    title="Show filters"
                ),
            ],
            className="d-flex align-items-center",
        ),
        className="bg-white shadow-sm",
    )

    # --- Search input row ---
    search_row = html.Div(
        [
            dbc.Input(
                id="search-input",
                placeholder="Search rules …",
                debounce=True, autoComplete="off",
                className="search-input-simple me-2 flex-grow-1",
            ),
            dbc.Button(
                DashIconify(icon="mdi:close"),
                id="clear-search",
                color="outline-secondary", size="sm",
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
        className=(
            "d-flex flex-column flex-md-row align-items-start "
            "align-items-md-center mb-3"
        ),
    )

    # --- Results container with loading spinner ---
    results_container = dcc.Loading(
        id="search-loading",
        type="circle",
        children=[
            html.Div(id="search-stats", className="mb-3"),
            html.Div(id="search-results"),
        ],
    )

    # --- Modal for rule detail view ---
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

    # --- Wrap in a Div so modal is outside the Card ---
    return html.Div([
        dbc.Card([header, dbc.CardBody([search_row, results_container]), offcanvas],
                 className="h-100 shadow-sm"),
        modal
    ])


# ---------------------------------------------------------------------------
# Result card builder
# ---------------------------------------------------------------------------

def make_rule_card(rule: dict, score: Optional[float], q_terms: List[str]) -> html.Div:
    """
    Build a single draggable rule card for display in results.
    
    Args:
        rule: Rule dict from retriever.
        score: Normalized search score (0–1), or None.
        q_terms: Query terms (unused yet, could support highlighting).
    """

    def badge(icon, label, color="secondary", extra_class=""):
        return dbc.Badge(
            [DashIconify(icon=icon, className="me-1"), label],
            color=color,
            className=extra_class
        )

    def tag_count_badge(icon, tags, color, label):
        count = len(tags)
        return badge(icon, f"{count} {label}", color, "me-2") if count else None

    card = dbc.Card(
        dbc.CardBody(
            [
                # --- Title + similarity badge ---
                html.Div(
                    [
                        html.H6(
                            str(rule.get("rule_name", "Untitled Rule")),
                            className="fw-bold mb-2 flex-grow-1"
                        ),
                        dbc.Badge(
                            [DashIconify(icon="mdi:target", className="me-1"),
                             f"{score:.1%}" if score is not None else ""],
                            color="success",
                            className="ms-auto" if score is not None else "d-none"
                        ),
                    ],
                    className="d-flex align-items-start",
                ),

                # --- Description text ---
                html.Div(
                    str(rule.get("rule_description", "No description")),
                    className="text-muted mb-3"
                ),

                # --- Metadata tags ---
                html.Div(
                    [
                        tag_count_badge("mdi:label", rule.get("rule_type"), "primary", "type(s)"),
                        tag_count_badge("mdi:earth", rule.get("country"), "info", "country(s)"),
                        tag_count_badge("mdi:account-group", rule.get("business_type"), "secondary", "team(s)"),
                        tag_count_badge("mdi:account-tie", rule.get("party_agent"), "warning", "agent(s)"),
                    ],
                    className="mb-2 d-flex flex-wrap"
                ),

                # --- Drag handle icon ---
                DashIconify(icon="mdi:drag", className="drag-handle position-absolute top-0 end-0 p-2"),
            ]
        ),
        className="rule-card shadow-sm"
    )

    return html.Div(
        card,
        id={"type": "rule-card", "index": rule["rule_id"]},
        className="mb-2 rule-card rule-card-clickable",
        n_clicks=0,
        draggable="true",
        style={"cursor": "grab"},
        **{"data-rule": json.dumps(rule)}  # Embed rule for detail modal
    )


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_search_callbacks(app, retriever):
    """
    Register all Dash callbacks for the search component.
    Called once from parent layout factory.

    Args:
        app: Dash app instance.
        retriever: Search retriever implementing `.search_rules()` and `.filter_options`.
    """

    # --- Populate filter dropdowns when opening sidebar ---
    @app.callback(
        Output("rule-type-filter", "options"),
        Output("country-filter", "options"),
        Output("business-team-filter", "options"),
        Output("party-agent-filter", "options"),
        Input("filters-offcanvas", "is_open"),
    )
    def _populate_filters(is_open):
        opts = retriever.filter_options if retriever else {}
        def fmt(keywords):
            return [{"label": k.title(), "value": k} for k in keywords]
        return (
            fmt(opts.get("rule_type", [])),
            fmt(opts.get("country", [])),
            fmt(opts.get("business_type", [])),
            fmt(opts.get("party_agent", [])),
        )

    # --- Toggle filter sidebar ---
    @app.callback(
        Output("filters-offcanvas", "is_open"),
        Input("open-filters", "n_clicks"),
        State("filters-offcanvas", "is_open")
    )
    def _toggle_filters(n, is_open):
        return not is_open if n else is_open

    # --- Clear search input ---
    @app.callback(
        Output("search-input", "value", allow_duplicate=True),
        Input("clear-search", "n_clicks"),
        prevent_initial_call=True
    )
    def _clear_input(n):
        return "" if n else ""

    # --- Clear all filters ---
    @app.callback(
        Output("rule-type-filter", "value"),
        Output("country-filter", "value"),
        Output("business-team-filter", "value"),
        Output("party-agent-filter", "value"),
        Input("clear-filters-btn", "n_clicks"),
        prevent_initial_call=True
    )
    def _clear_filters(n):
        return None, None, None, None

    # --- Main search (runs automatically on any input change) ---
    @app.callback(
        [Output("search-results", "children"), Output("search-stats", "children")],
        [Input("search-input", "value"),
         Input("rule-type-filter", "value"),
         Input("country-filter", "value"),
         Input("business-team-filter", "value"),
         Input("party-agent-filter", "value"),
         Input("exact-switch", "value"),
         Input("rule-type-filter", "options"),
         Input("country-filter", "options"),
         Input("business-team-filter", "options"),
         Input("party-agent-filter", "options")]
    )
    def _search(query, rule_type, country, business_type, party_agent, exact,
                rule_type_opts, country_opts, business_type_opts, party_agent_opts):

        if retriever is None:
            logger.error("Search attempted but retriever is not initialized")
            return [html.Div("Search backend not initialised", className="text-danger")], None

        mode = SearchMode.KEYWORD if exact else SearchMode.HYBRID

        if not (query or rule_type or country or business_type or party_agent):
            logger.debug("Empty search: no query or filters applied")
            empty = html.Div([
                DashIconify(icon="mdi:file-search-outline", width=48, className="text-muted mb-3"),
                html.P("Start typing to search rules …", className="text-muted"),
            ], className="text-center py-5")
            return empty, ""

        start = time.time()
        try:
            from config import DEFAULT_SEARCH_LIMIT
            logger.info(
                "Executing search: query=%r, mode=%s, filters=%s",
                query, mode.name, {"rule_type": rule_type, "country": country,
                                   "business_type": business_type, "party_agent": party_agent}
            )

            rows = retriever.search_rules(
                query=query,
                rule_type=rule_type,
                country=country,
                business_type=business_type,
                party_agent=party_agent,
                mode=mode,
                top_k=DEFAULT_SEARCH_LIMIT
            )

            def prep_scores(rows, mode):
                vals = [float(r.get("search_score", 0)) for r in rows] or [0.0]
                if mode == SearchMode.HYBRID:
                    norm = lambda x: max(0.0, min(1.0, float(x)))
                else:
                    lo, hi = min(vals), max(vals)
                    norm = lambda x: 0.0 if hi == lo else (float(x) - lo) / (hi - lo)
                for r in rows:
                    r["search_score"] = norm(float(r.get("search_score", 0))) if r.get("search_score") is not None else None
                return rows

            elapsed = time.time() - start

            if not rows:
                logger.info("No matches found (%.3fs)", elapsed)
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
            rows = prep_scores(rows, mode)
            cards = [make_rule_card(r, r.get("search_score"), q_terms) for r in rows]

            logger.info("Search returned %d matches in %.3fs", len(cards), elapsed)
            stats = dbc.Alert([
                DashIconify(icon="mdi:check-circle", className="me-2"),
                f"{len(cards)} rule(s) · {elapsed:.3f}s"
            ], color="success", className="mb-3")
            return cards, stats

        except Exception as exc:
            logger.exception("Search failed: %s", exc)
            err_view = html.Div([
                DashIconify(icon="mdi:alert-circle", width=48, className="text-danger mb-3"),
                html.P(f"Search error: {exc}", className="text-danger"),
            ], className="text-center py-5")
            alert = dbc.Alert([
                DashIconify(icon="mdi:alert", className="me-2"),
                f"Search failed: {exc}"
            ], color="danger", className="mb-0")
            return err_view, alert

    # --- Display rule detail modal on card click ---
    @app.callback(
        Output("rule-detail-modal", "is_open"),
        Output("rule-modal-body", "children"),
        Output({"type": "rule-card", "index": ALL}, "n_clicks"),
        Input({"type": "rule-card", "index": ALL}, "n_clicks"),
        State({"type": "rule-card", "index": ALL}, "data-rule"),
        State("rule-detail-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _display_rule_details(clicks, rules_data, is_open):
        if not any(clicks):
            return is_open, no_update, [0] * len(clicks)

        clicked_idx = next(i for i, c in enumerate(clicks) if c)
        rule = json.loads(rules_data[clicked_idx])
        logger.debug("Opening detail modal for rule_id=%s", rule.get("rule_id"))

        # --- Fields shown in detail modal ---
        FIELD_ORDER = [
            "description_en", "description_de", "rule_code", "bansta_error_code",
            "iso_error_code", "llm_description", "keywords", "rule_type", "country",
            "business_type", "party_agent", "relevance"
        ]
        DISPLAY_NAMES = {
            "rule_name": "Rule Name",
            "rule_description": "Rule Description",
            "bansta_error_code": "BANSTA Error Code",
            "iso_error_code": "ISO Error Code",
            "description_en": "Description",
            "description_de": "Beschreibung",
            "rule_code": "Code",
            "llm_description": "LLM Description",
            "keywords": "Keywords",
            "rule_type": "Rule Type",
            "country": "Country",
            "business_type": "Business Type",
            "party_agent": "Party Agent",
            "version_major": "Version Major",
            "version_minor": "Version Minor",
        }

        rows = []
        for k in FIELD_ORDER:
            val = rule.get(k)
            if val in [None, "", []]:
                continue
            label = DISPLAY_NAMES.get(k, k.replace("_", " ").title())

            if k == "rule_code":
                rows.append(
                    html.Div([
                        html.H6(label, className="fw-bold mb-1"),
                        html.Pre(
                            str(val),
                            className="mb-3 p-2 rounded border border-1",
                            style={
                                "fontFamily": "monospace",
                                "whiteSpace": "pre-wrap",
                                "backgroundColor": "#e3f2fd",
                                "borderLeft": "5px solid #1976d2",
                                "color": "#1565c0",
                            }
                        ),
                    ], className="mb-3")
                )
            elif k in ["rule_type", "country", "business_type", "party_agent"]:
                tags = val if isinstance(val, (list, set, tuple)) else [val]
                tags = [str(tag) for tag in tags if tag and tag != "null"]
                icon_map = {
                    "rule_type": "mdi:label",
                    "country": "mdi:earth",
                    "business_type": "mdi:account-group",
                    "party_agent": "mdi:account-tie",
                }
                color_map = {
                    "rule_type": "primary",
                    "country": "info",
                    "business_type": "secondary",
                    "party_agent": "warning",
                }
                rows.append(
                    html.Div([
                        html.H6(label, className="fw-bold mb-1"),
                        html.Div([
                            dbc.Badge([
                                DashIconify(icon=icon_map[k], className="me-1"),
                                html.Span(tag, className="ps-1 pe-2"),
                            ], color=color_map[k],
                               className="me-2 mb-2 shadow-sm",
                               style={"fontSize": "1em"})
                            for tag in tags
                        ], className="mb-3 d-flex flex-wrap gap-1"),
                    ], className="mb-3")
                )
            else:
                rows.append(
                    html.Div([
                        html.H6(label, className="fw-bold mb-1"),
                        html.Div(str(val), className="mb-3 text-muted ps-2"),
                    ], className="mb-3")
                )

        return True, rows, [0] * len(clicks)


__all__ = ["register_search_callbacks", "create_search_component"]
