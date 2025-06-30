import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc

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
server = app.server

# Set layout
app.layout = create_layout()

# Register callbacks
register_search_callbacks(app)
register_generator_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
