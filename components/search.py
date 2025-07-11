from dash import html, dcc, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import json
import time
from typing import Optional

from rag.retriever import RuleRetriever, SearchMode, SearchConfig
from rag.embeddings import EmbeddingManager

# Initialize retriever (add this at module level)
_retriever = None

def initialize_retriever(rules_path: Optional[str] = None):
    """Initialize the global retriever - call this once at app startup"""
    global _retriever
    
    config = SearchConfig(
        semantic_weight=0.7,
        bm25_weight=0.2,
        fuzzy_weight=0.1,
        min_similarity=0.15,
        enable_reranking=True
    )
    
    embedding_manager = EmbeddingManager(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="./embedding_cache"
    )
    
    _retriever = RuleRetriever(embedding_manager, config)
    
    if rules_path:
        _retriever.load_rules(rules_path)

def create_search_component():
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H4([
                    DashIconify(icon="mdi:magnify", className="me-2"),
                    "Search Rules"
                ], className="mb-0 fw-semibold"),
                
                # Search mode toggle
                dbc.ButtonGroup([
                    dbc.Button([
                        DashIconify(icon="mdi:brain", className="me-1"),
                        "Smart"
                    ], id="mode-hybrid", color="primary", size="sm", active=True),
                    dbc.Button([
                        DashIconify(icon="mdi:text-search", className="me-1"),
                        "Keyword"
                    ], id="mode-keyword", color="outline-primary", size="sm"),
                    dbc.Button([
                        DashIconify(icon="mdi:vector-triangle", className="me-1"),
                        "Semantic"
                    ], id="mode-semantic", color="outline-primary", size="sm")
                ], className="ms-auto")
            ], className="d-flex align-items-center")
        ]),
        
        dbc.CardBody([
            # Search controls
            html.Div([
                # Search input with clear button
                dbc.InputGroup([
                    dbc.Input(
                        id="search-input",
                        placeholder="Search rules... (e.g., 'null validation', 'password requirements')",
                        className="border-end-0",
                        debounce=True
                    ),
                    dbc.Button([
                        DashIconify(icon="mdi:magnify", width=20)
                    ], id="search-btn", color="primary", outline=True),
                    dbc.Button([
                        DashIconify(icon="mdi:close", width=16)
                    ], id="clear-search", color="outline-secondary", size="sm")
                ], className="mb-3"),
                
                # Filters with results limit
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            id="category-filter",
                            placeholder="Category",
                            options=[
                                {"label": "Business Rules", "value": "business"},
                                {"label": "Technical Rules", "value": "technical"},
                                {"label": "Compliance", "value": "compliance"},
                                {"label": "Security", "value": "security"},
                                {"label": "Validation", "value": "validation"}
                            ],
                            className="filter-dropdown"
                        )
                    ], width=4),
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
                    ], width=4),
                    dbc.Col([
                        dcc.Dropdown(
                            id="results-limit",
                            placeholder="Results",
                            options=[
                                {"label": "5 results", "value": 5},
                                {"label": "10 results", "value": 10},
                                {"label": "20 results", "value": 20},
                                {"label": "50 results", "value": 50}
                            ],
                            value=10,
                            className="filter-dropdown"
                        )
                    ], width=4)
                ], className="mb-3"),
                
                # Quick search suggestions
                html.Div([
                    html.P("Quick searches:", className="mb-2 text-muted small"),
                    dbc.Badge("null validation", id="quick-null", color="light", 
                             text_color="primary", className="me-2 mb-1 cursor-pointer"),
                    dbc.Badge("password security", id="quick-password", color="light", 
                             text_color="primary", className="me-2 mb-1 cursor-pointer"),
                    dbc.Badge("transaction limits", id="quick-transaction", color="light", 
                             text_color="primary", className="me-2 mb-1 cursor-pointer"),
                    dbc.Badge("email format", id="quick-email", color="light", 
                             text_color="primary", className="me-2 mb-1 cursor-pointer")
                ], className="mb-4"),
                
                # Search stats
                html.Div(id="search-stats", className="mb-3")
            ]),
            
            # Results area with loading
            dcc.Loading(
                id="search-loading",
                children=html.Div(id="search-results", className="search-results"),
                type="circle"
            )
        ])
    ], className="h-100 search-card")

def create_rule_card(rule, similarity_score=None):
    """Enhanced rule card with similarity score"""
    
    priority_colors = {"high": "danger", "medium": "warning", "low": "success"}
    category_colors = {
        "business": "primary", "technical": "info", "compliance": "secondary",
        "security": "danger", "validation": "warning"
    }
    
    priority_color = priority_colors.get(rule.get("priority", "low"), "secondary")
    category_color = category_colors.get(rule.get("category", "business"), "primary")
    
    return html.Div([
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    # Header with title and score
                    html.Div([
                        html.H6(rule.get("name", "Untitled Rule"), 
                               className="fw-bold mb-2 rule-title"),
                        dbc.Badge([
                            DashIconify(icon="mdi:target", className="me-1"),
                            f"{similarity_score:.1%}"
                        ], color="success", className="ms-auto") if similarity_score else None
                    ], className="d-flex justify-content-between align-items-start"),
                    
                    # Description
                    html.P(rule.get("description", "No description"), 
                          className="text-muted mb-3 rule-desc"),
                    
                    # Badges
                    html.Div([
                        dbc.Badge([
                            DashIconify(icon="mdi:folder", className="me-1"),
                            rule.get("category", "general").title()
                        ], color=category_color, className="me-2"),
                        dbc.Badge([
                            DashIconify(icon="mdi:priority-high", className="me-1"),
                            rule.get("priority", "medium").title()
                        ], color=priority_color, className="me-2")
                    ], className="mb-2"),
                    
                    # Drag handle
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
    # Search mode selection
    @app.callback(
        [Output("mode-hybrid", "active"),
         Output("mode-keyword", "active"), 
         Output("mode-semantic", "active"),
         Output("mode-hybrid", "color"),
         Output("mode-keyword", "color"),
         Output("mode-semantic", "color")],
        [Input("mode-hybrid", "n_clicks"),
         Input("mode-keyword", "n_clicks"),
         Input("mode-semantic", "n_clicks")]
    )
    def update_search_mode(hybrid_clicks, keyword_clicks, semantic_clicks):
        if not ctx.triggered:
            return True, False, False, "primary", "outline-primary", "outline-primary"
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "mode-hybrid":
            return True, False, False, "primary", "outline-primary", "outline-primary"
        elif button_id == "mode-keyword":
            return False, True, False, "outline-primary", "primary", "outline-primary"
        else:  # semantic
            return False, False, True, "outline-primary", "outline-primary", "primary"
    
    # Quick search clicks
    @app.callback(
        Output("search-input", "value"),
        [Input("quick-null", "n_clicks"),
         Input("quick-password", "n_clicks"),
         Input("quick-transaction", "n_clicks"),
         Input("quick-email", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_quick_search(null_clicks, password_clicks, transaction_clicks, email_clicks):
        if not ctx.triggered:
            return ""
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        quick_searches = {
            "quick-null": "null validation",
            "quick-password": "password security", 
            "quick-transaction": "transaction limits",
            "quick-email": "email format"
        }
        
        return quick_searches.get(button_id, "")
    
    # Clear search
    @app.callback(
        [Output("search-input", "value", allow_duplicate=True),
         Output("category-filter", "value"),
         Output("priority-filter", "value")],
        Input("clear-search", "n_clicks"),
        prevent_initial_call=True
    )
    def clear_search(n_clicks):
        if n_clicks:
            return "", None, None
        return "", None, None
    
    # Main search callback
    @app.callback(
        [Output("search-results", "children"),
         Output("search-stats", "children")],
        [Input("search-btn", "n_clicks"),
         Input("search-input", "n_submit"),
         Input("search-input", "value")],  # Added for live search
        [State("category-filter", "value"),
         State("priority-filter", "value"),
         State("results-limit", "value"),
         State("mode-hybrid", "active"),
         State("mode-keyword", "active"),
         State("mode-semantic", "active")]
    )
    def update_search_results(n_clicks, n_submit, query, category, priority, limit, 
                            hybrid_active, keyword_active, semantic_active):
        
        # Determine search mode
        if hybrid_active:
            search_mode = SearchMode.HYBRID
        elif keyword_active:
            search_mode = SearchMode.KEYWORD
        elif semantic_active:
            search_mode = SearchMode.SEMANTIC
        else:
            search_mode = SearchMode.HYBRID
        
        # Empty state
        if not query and not category and not priority:
            return (
                html.Div([
                    DashIconify(icon="mdi:file-search-outline", width=48, className="text-muted mb-3"),
                    html.P("Start typing to search rules...", className="text-muted")
                ], className="text-center py-5"),
                ""
            )
        
        # Perform search
        start_time = time.time()
        
        try:
            if _retriever is None:
                # Fallback to original function if retriever not initialized
                from rag.retriever import search_rules
                results = search_rules(query, category, priority)
                results_df = None
            else:
                results_df = _retriever.search_rules(
                    query=query,
                    category=category, 
                    priority=priority,
                    mode=search_mode,
                    top_k=limit or 10
                )
                results = results_df.to_dict('records') if len(results_df) > 0 else []
            
            search_time = time.time() - start_time
            
            # No results
            if not results:
                return (
                    html.Div([
                        DashIconify(icon="mdi:file-remove-outline", width=48, className="text-muted mb-3"),
                        html.P("No rules found matching your criteria", className="text-muted")
                    ], className="text-center py-5"),
                    dbc.Alert([
                        DashIconify(icon="mdi:information", className="me-2"),
                        f"Search completed in {search_time:.3f}s - No matches found"
                    ], color="warning", className="mb-0")
                )
            
            # Create result cards
            result_cards = []
            for rule in results:
                similarity_score = rule.get('search_score') if results_df is not None else None
                result_cards.append(create_rule_card(rule, similarity_score))
            
            # Create stats
            stats_text = f"Found {len(results)} rule(s)"
            if query:
                stats_text += f" for '{query}'"
            stats_text += f" in {search_time:.3f}s"
            
            stats = dbc.Alert([
                DashIconify(icon="mdi:check-circle", className="me-2"),
                stats_text
            ], color="success", className="mb-3")
            
            return result_cards, stats
            
        except Exception as e:
            return (
                html.Div([
                    DashIconify(icon="mdi:alert-circle", width=48, className="text-danger mb-3"),
                    html.P(f"Search error: {str(e)}", className="text-danger")
                ], className="text-center py-5"),
                dbc.Alert([
                    DashIconify(icon="mdi:alert", className="me-2"),
                    f"Search failed: {str(e)}"
                ], color="danger", className="mb-0")
            )