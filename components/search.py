from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import json

from rag.retriever import search_rules

def create_search_component():
    return dbc.Card([
        dbc.CardHeader([
            html.H4([
                DashIconify(icon="mdi:magnify", className="me-2"),
                "Search Rules"
            ], className="mb-0 fw-semibold")
        ]),
        
        dbc.CardBody([
            # Search controls
            html.Div([
                # Search input
                dbc.InputGroup([
                    dbc.Input(
                        id="search-input",
                        placeholder="Search rules...",
                        className="border-end-0"
                    ),
                    dbc.Button([
                        DashIconify(icon="mdi:magnify", width=20)
                    ], id="search-btn", color="primary", outline=True)
                ], className="mb-3"),
                
                # Filters
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id="category-filter",
                            placeholder="Category",
                            options=[
                                {"label": "Business Rules", "value": "business"},
                                {"label": "Technical Rules", "value": "technical"},
                                {"label": "Compliance", "value": "compliance"}
                            ],
                            className="filter-dropdown"
                        )
                    ], width=6),
                    dbc.Col([
                        dcc.Dropdown(
                            id="priority-filter", 
                            placeholder="Priority",
                            options=[
                                {"label": "High", "value": "high"},
                                {"label": "Medium", "value": "medium"},
                                {"label": "Low", "value": "low"}
                            ],
                            className="filter-dropdown"
                        )
                    ], width=6)
                ], className="mb-4")
            ]),
            
            # Results area
            html.Div(id="search-results", className="search-results")
        ])
    ], className="h-100 search-card")


def create_rule_card(rule):
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6(rule["name"], className="fw-bold mb-2 rule-title"),
                    html.P(rule["description"], className="text-muted mb-2 rule-desc"),
                    dbc.Badge(rule["category"], color="primary", className="me-2"),
                    dbc.Badge(rule["priority"], color="secondary"),
                    html.Div([
                        DashIconify(icon="mdi:drag", className="drag-handle")
                    ], className="position-absolute top-0 end-0 p-2")
                ])
            ])
        ])
    ], 
    className="rule-card mb-2", 
    draggable="true",
    **{"data-rule": json.dumps(rule)}
)

def register_search_callbacks(app):
    @app.callback(
        Output("search-results", "children"),
        [Input("search-btn", "n_clicks"),
         Input("search-input", "n_submit")],
        [State("search-input", "value"),
         State("category-filter", "value"),
         State("priority-filter", "value")]
    )
    def update_search_results(n_clicks, n_submit, query, category, priority):
        if not query and not category and not priority:
            return html.Div([
                DashIconify(icon="mdi:file-search-outline", width=48, className="text-muted mb-3"),
                html.P("Start typing to search rules...", className="text-muted")
            ], className="text-center py-5")
        
        # Get search results
        results = search_rules(query, category, priority)
        
        if not results:
            return html.Div([
                DashIconify(icon="mdi:file-remove-outline", width=48, className="text-muted mb-3"),
                html.P("No rules found matching your criteria", className="text-muted")
            ], className="text-center py-5")
        
        return [create_rule_card(rule) for rule in results]