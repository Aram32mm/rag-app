from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

from .search import create_search_component
from .generator import create_generator_component

def create_layout():
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.Div([
                    DashIconify(icon="mdi:brain", width=32, height=32, className="me-3"),
                    html.H1("RAG Rules Engine", className="mb-0 fw-bold")
                ], className="d-flex align-items-center py-3")
            ])
        ], className="border-bottom mb-4"),
        
        # Main content
        dbc.Row([
            # Search component - Left side
            dbc.Col([
                create_search_component()
            ], width=6, className="pe-3"),
            
            # Generator component - Right side  
            dbc.Col([
                create_generator_component()
            ], width=6, className="ps-3")
        ], className="h-100")
        
    ], fluid=True, className="main-container")