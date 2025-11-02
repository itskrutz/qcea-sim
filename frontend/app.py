import dash
import dash_bootstrap_components as dbc

# Import layout and callback functions from other files in this package
# The '.' means "from the current package"
from .layout import build_layout
from .callbacks import register_callbacks

# Initialize the Dash app
# Dash automatically serves files from a folder named 'assets'
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "QCEA-Sim Dashboard"

# Set the app layout by calling the function from layout.py
app.layout = build_layout()

# Register all callbacks by calling the function from callbacks.py
register_callbacks(app)