import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go
import networkx as nx

# --- Define colors from CSS for plots ---
PLOT_BG_COLOR = '#1D1D1D'
PLOT_FONT_COLOR = '#F1F5F9'
PQCEA_COLOR = '#E040FB'      # Vibrant Magenta
QCEA_COLOR = '#BEF264'       # Lime Green
DIJKSTRA_COLOR = '#38BDF8'   # Light Blue
GRID_COLOR = '#333333'
# ------------------------------------------------

# --- Plot & Card Creation Functions ---

def create_network_figure(G=None, active_paths=[]):
    """Generates the Plotly Figure for the NetworkX graph with link utilization."""
    if G is None:
        return create_empty_figure("Live Network Topology")
        
    pos = nx.spring_layout(G, seed=42)
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    
    has_energy = 'residual_energy_j' in next(iter(G.nodes(data=True)))[1]

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        energy_text = "N/A"
        energy_frac = 0.5 
        if has_energy:
            energy_val = data.get('residual_energy_j', 0)
            energy_text = f"{energy_val:.0f} J"
            max_energy = 1e6 if data.get('role') == 'core' else 1e4
            energy_frac = energy_val / (max_energy + 1e-9)

        node_text.append(f"<b>Node {node}</b><br>Role: {data.get('role', 'N/A')}<br>Energy: {energy_text}")
        
        if energy_frac > 0.7: node_color.append('#22C55E') # Green
        elif energy_frac > 0.3: node_color.append('#F59E0B') # Amber
        else: node_color.append('#EF4444') # Red

        node_size.append(20 if data.get('role') == 'core' else 12)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(color=node_color, size=node_size, line_width=2, line_color='white')
    )

    # Enhanced edge visualization with utilization
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Get utilization (0.0 to 1.0)
        util = data.get('utilization', 0.0)
        
        # Color based on utilization: green -> yellow -> red
        if util < 0.3:
            edge_color = '#22C55E'  # Green (low)
            edge_width = 1.5
        elif util < 0.7:
            edge_color = '#F59E0B'  # Yellow (medium)
            edge_width = 2
        else:
            edge_color = '#EF4444'  # Red (high/congested)
            edge_width = 3
        
        # Create hover text with link info
        bw_mbps = data.get('bandwidth_bps', 0) / 1e6
        delay = data.get('prop_delay_ms', 0)
        loss = data.get('loss_prob', 0)
        flows = data.get('active_flows', 0)
        
        hover_text = (f"Link {u}-{v}<br>"
                     f"Utilization: {util*100:.1f}%<br>"
                     f"Bandwidth: {bw_mbps:.1f} Mbps<br>"
                     f"Delay: {delay:.2f} ms<br>"
                     f"Loss: {loss*100:.2f}%<br>"
                     f"Active Flows: {flows}")
        
        edge_trace = go.Scatter(
            x=[x0, x1], y=[y0, y1],
            line=dict(width=edge_width, color=edge_color),
            hoverinfo='text',
            text=hover_text,
            mode='lines'
        )
        edge_traces.append(edge_trace)
    
    # Active path visualization with animation effect
    path_traces = []
    path_colors = [PQCEA_COLOR, QCEA_COLOR, DIJKSTRA_COLOR, '#22C55E', '#F59E0B']
    for i, path in enumerate(active_paths):
        path_x, path_y = [], []
        color = path_colors[i % len(path_colors)]
        for node in path:
            x, y = pos[node]
            path_x.append(x)
            path_y.append(y)
        path_traces.append(go.Scatter(
            x=path_x, y=path_y, 
            line=dict(width=5, color=color),
            mode='lines+markers',
            marker=dict(size=8, color=color, symbol='circle'),
            name=f"Active Path {i+1}",
            hoverinfo='skip'
        ))

    fig = go.Figure(
        data=edge_traces + [node_trace] + path_traces,
        layout=go.Layout(
            title='Live Network Topology',
            showlegend=len(path_traces) > 0,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=PLOT_BG_COLOR,
            paper_bgcolor=PLOT_BG_COLOR,
            font_color=PLOT_FONT_COLOR,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
    )
    return fig

def create_empty_figure(title):
    """Creates a blank figure with a title, styled for the theme."""
    return go.Figure(
        layout=go.Layout(
            title=title,
            xaxis=dict(title='Time Step', gridcolor=GRID_COLOR, zeroline=False), 
            yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
            paper_bgcolor=PLOT_BG_COLOR,
            plot_bgcolor=PLOT_BG_COLOR,
            font_color=PLOT_FONT_COLOR
        )
    )

def create_kpi_card(title, value, color):
    """Creates a single KPI stat card using the custom CSS class."""
    return html.Div(
        [
            html.P(title, className="card-title"),
            html.H3(value, className="card-text", style={'color': color}),
        ],
        className="kpi-card"
    )

# --- Layout Building Functions ---

def build_navbar():
    """Builds the top navigation bar."""
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Span("🚀", style={'fontSize': '1.5rem'})),
                        dbc.Col(
                            html.Div([
                                html.Span("P-QCEA"),
                                html.Span(" // Predictive Simulator", className="navbar-brand-accent")
                            ],
                            className="navbar-brand",
                            )
                        ),
                    ],
                    align="center",
                    className="g-0", # No gutters
                ),
                href="#",
                style={"textDecoration": "none"},
            )
        ], fluid=True),
        className="mb-4",
    )


def build_control_panel():
    """Builds the collapsible control panel."""
    return dbc.Card(
        dbc.CardBody([
            # --- Topology & Traffic ---
            dbc.Row([
                dbc.Col(html.H5("Simulation Parameters"), width=12),
                dbc.Col([
                    dbc.Label("Simulation Time Steps:"),
                    dcc.Slider(id="slider-time-steps", min=10, max=200, step=10, value=100, marks={i: str(i) for i in range(0, 201, 50)}),
                ], width=4),
                dbc.Col([
                    dbc.Label("Number of Nodes:"),
                    dcc.Slider(id="slider-nodes", min=10, max=100, step=5, value=25, marks={i: str(i) for i in range(10, 101, 20)}),
                ], width=4),
                dbc.Col([
                    dbc.Label("Number of Flows (per step):"),
                    dbc.Input(id="input-flows", type="number", value=5, min=1, max=50),
                ], width=4),
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Random Seed (for reproducibility):"),
                    dbc.Input(id="input-seed", type="number", value=42, min=0, max=9999),
                    html.Small("Same seed = same results", style={'color': 'var(--muted-text)', 'fontStyle': 'italic'}),
                ], width=4),
            ], className="mt-2"),
            
            # --- Traffic Mix Control (NEW) ---
            html.Hr(style={'borderColor': GRID_COLOR}),
            dbc.Row([
                dbc.Col(html.H5("Traffic Mix Distribution"), width=12),
                dbc.Col([
                    dbc.Label("VoIP (Real-time voice):"),
                    dcc.Slider(id="slider-voip-mix", min=0, max=100, step=5, value=33, 
                               marks={i: f"{i}%" for i in range(0, 101, 25)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], width=4),
                dbc.Col([
                    dbc.Label("Video (Streaming):"),
                    dcc.Slider(id="slider-video-mix", min=0, max=100, step=5, value=33,
                               marks={i: f"{i}%" for i in range(0, 101, 25)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], width=4),
                dbc.Col([
                    dbc.Label("Best Effort (Data):"),
                    dcc.Slider(id="slider-data-mix", min=0, max=100, step=5, value=34,
                               marks={i: f"{i}%" for i in range(0, 101, 25)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], width=4),
            ]),
            html.Div(
                html.Small("Note: Percentages will be normalized if they don't sum to 100%", 
                          style={'color': 'var(--muted-text)', 'fontStyle': 'italic'}),
                className="mt-2"
            ),
            html.Hr(style={'borderColor': GRID_COLOR}),
            
            # --- Routing Algorithm Weights ---
            dbc.Row([
                dbc.Col(html.H5("Routing Algorithm"), width=12),
                dbc.Col(
                    dcc.RadioItems(
                        id="radio-algo",
                        options=[
                            {'label': ' P-QCEA (Predictive)', 'value': 'pqcea'},
                            {'label': ' QCEA (Reactive)', 'value': 'qcea'},
                            {'label': ' Dijkstra (Baseline)', 'value': 'dijkstra'},
                        ],
                        value='pqcea', 
                        labelStyle={'display': 'block', 'margin-bottom': '10px'},
                        inputClassName="me-2"
                    ), width=3
                ),
                # Weight configuration panel
                dbc.Col(
                    html.Div(id="qcea-weights-panel", children=[
                        dbc.Alert([
                            html.Strong("💡 Weight Configuration"),
                            html.Br(),
                            html.Small([
                                "Weights are automatically applied per traffic type, but you can override for experiments:",
                                html.Br(),
                                "• VoIP uses wl=0.5 (latency priority) | ",
                                "• Video uses wb=0.5 (bandwidth) | ",
                                "• Data uses we=0.4 (energy)"
                            ])
                        ], color="info", style={'fontSize': '0.75rem', 'marginBottom': '0.5rem'}),
                        html.Div([
                            html.B("Mode: ", style={'fontSize': '0.85rem'}),
                            dcc.RadioItems(
                                id='radio-weight-mode',
                                options=[
                                    {'label': ' Auto (Per-Traffic)', 'value': 'auto'},
                                    {'label': ' Manual (Global)', 'value': 'manual'},
                                ],
                                value='auto',
                                inline=True,
                                style={'fontSize': '0.85rem'}
                            )
                        ], style={'marginTop': '0.5rem'}),
                        html.Div(id='manual-weight-sliders', children=[
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("wl (Latency):", style={'fontSize': '0.85rem'}),
                                    dcc.Slider(id="slider-wl", min=0, max=1, step=0.05, value=0.3, 
                                              marks={0: '0', 0.5: '0.5', 1: '1'}, 
                                              tooltip={"placement": "bottom", "always_visible": True}),
                                    dbc.Label("wb (Bandwidth):", className="mt-2", style={'fontSize': '0.85rem'}),
                                    dcc.Slider(id="slider-wb", min=0, max=1, step=0.05, value=0.2, 
                                              marks={0: '0', 0.5: '0.5', 1: '1'}, 
                                              tooltip={"placement": "bottom", "always_visible": True}),
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("wp (Packet Loss):", style={'fontSize': '0.85rem'}),
                                    dcc.Slider(id="slider-wp", min=0, max=1, step=0.05, value=0.2, 
                                              marks={0: '0', 0.5: '0.5', 1: '1'}, 
                                              tooltip={"placement": "bottom", "always_visible": True}),
                                    dbc.Label("we (Energy):", className="mt-2", style={'fontSize': '0.85rem'}),
                                    dcc.Slider(id="slider-we", min=0, max=1, step=0.05, value=0.2, 
                                              marks={0: '0', 0.5: '0.5', 1: '1'}, 
                                              tooltip={"placement": "bottom", "always_visible": True}),
                                    dbc.Label("wc (Cost):", className="mt-2", style={'fontSize': '0.85rem'}),
                                    dcc.Slider(id="slider-wc", min=0, max=1, step=0.05, value=0.1, 
                                              marks={0: '0', 0.5: '0.5', 1: '1'}, 
                                              tooltip={"placement": "bottom", "always_visible": True}),
                                ], width=6),
                            ], style={'marginTop': '1rem'})
                        ], style={'display': 'none'})
                    ]), width=9
                ),
            ]),
            html.Hr(style={'borderColor': GRID_COLOR}),
            
            # --- Prediction Parameters ---
            html.Div(id="prediction-params-panel", children=[
                dbc.Row([
                    dbc.Col(html.H5("Prediction Parameters (for P-QCEA)"), width=12),
                    dbc.Col([
                        dbc.Label("Prediction Horizon (steps):"),
                        dcc.Slider(id="slider-horizon", min=1, max=10, step=1, value=3, 
                                   marks={i: str(i) for i in range(1, 11, 2)}, className="prediction-slider"),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Smoothing (Alpha):"),
                        dcc.Slider(id="slider-alpha", min=0.1, max=1.0, step=0.1, value=0.3, 
                                   marks={i/10: f"{i/10:.1f}" for i in range(1, 11, 2)}, className="prediction-slider"),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Confidence Threshold:"),
                        dcc.Slider(id="slider-confidence", min=0.1, max=1.0, step=0.1, value=0.7, 
                                   marks={i/10: f"{i/10:.1f}" for i in range(1, 11, 2)}, className="prediction-slider"),
                    ], width=4),
                ])
            ]),
        ]),
    )

def build_layout():
    """Builds the main app layout."""
    return html.Div([
        dcc.Store(id='store-sim-data-dijkstra', data={'G_json': None, 'metrics': []}),
        dcc.Store(id='store-sim-data-qcea', data={'G_json': None, 'metrics': []}),
        dcc.Store(id='store-sim-data-pqcea', data={'G_json': None, 'metrics': [], 'predictor_history': {}}), 
        dcc.Store(id='store-sim-state', data={'running': False, 'step': 0, 'max_steps': 0}),
        dcc.Interval(id='interval-sim-step', interval=500, n_intervals=0, disabled=True),
        
        build_navbar(),
        
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("P-QCEA Predictive Routing Simulator", style={'fontWeight': '700'}),
                    html.P("Comparing Proactive (P-QCEA) vs. Reactive (QCEA) vs. Baseline (Dijkstra)", className="lead", style={'color': 'var(--muted-text)'}),
                    html.Div("Status: Idle", id="sim-status", className="mt-2 mb-3", style={'color': 'var(--muted-text)', 'fontFamily': 'monospace'}),
                    dbc.Button("Simulation Controls", id="btn-toggle-controls", color="primary", className="me-2"),
                    dbc.Button("► Start", id="btn-start", color="success", n_clicks=0),
                    dbc.Button("■ Stop", id="btn-stop", color="danger", n_clicks=0, className="ms-2"),
                    dbc.Button("📊 Export Report", id="btn-export-report", color="info", className="ms-2", n_clicks=0),
                    dcc.Download(id="download-report"),
                    dcc.Store(id='store-report-status', data={'last_export': None}),
                ], width=12),
            ]),

            dbc.Collapse(build_control_panel(), id="collapse-controls", is_open=False),
            
            # --- 3x3 KPI Grid ---
            dbc.Row([
                # --- THIS IS THE FIX ---
                dbc.Col(html.H4("Avg. Path Latency (ms)", className="text-center kpi-grid-title mb-3"), width=12),
                # -----------------------
                dbc.Col(create_kpi_card("Dijkstra", "N/A", DIJKSTRA_COLOR), id="kpi-dijkstra-latency", width=4),
                dbc.Col(create_kpi_card("QCEA", "N/A", QCEA_COLOR), id="kpi-qcea-latency", width=4),
                dbc.Col(create_kpi_card("P-QCEA", "N/A", PQCEA_COLOR), id="kpi-pqcea-latency", width=4),
            ], className="mt-4"),
            
            dbc.Row([
                # --- THIS IS THE FIX ---
                dbc.Col(html.H4("Total Residual Energy (J)", className="text-center kpi-grid-title mb-3"), width=12),
                # -----------------------
                dbc.Col(create_kpi_card("Dijkstra", "N/A", DIJKSTRA_COLOR), id="kpi-dijkstra-energy", width=4),
                dbc.Col(create_kpi_card("QCEA", "N/A", QCEA_COLOR), id="kpi-qcea-energy", width=4),
                dbc.Col(create_kpi_card("P-QCEA", "N/A", PQCEA_COLOR), id="kpi-pqcea-energy", width=4),
            ]),
            
            dbc.Row([
                # --- THIS IS THE FIX ---
                dbc.Col(html.H4("Packet Delivery Rate (%)", className="text-center kpi-grid-title mb-3"), width=12),
                # -----------------------
                dbc.Col(create_kpi_card("Dijkstra", "N/A", DIJKSTRA_COLOR), id="kpi-dijkstra-pdr", width=4),
                dbc.Col(create_kpi_card("QCEA", "N/A", QCEA_COLOR), id="kpi-qcea-pdr", width=4),
                dbc.Col(create_kpi_card("P-QCEA", "N/A", PQCEA_COLOR), id="kpi-pqcea-pdr", width=4),
            ]),
            
            # NEW: Jitter Metric Row
            dbc.Row([
                dbc.Col(html.H4("Avg. Jitter (ms)", className="text-center kpi-grid-title mb-3"), width=12),
                dbc.Col(create_kpi_card("Dijkstra", "N/A", DIJKSTRA_COLOR), id="kpi-dijkstra-jitter", width=4),
                dbc.Col(create_kpi_card("QCEA", "N/A", QCEA_COLOR), id="kpi-qcea-jitter", width=4),
                dbc.Col(create_kpi_card("P-QCEA", "N/A", PQCEA_COLOR), id="kpi-pqcea-jitter", width=4),
            ]),
            
            # Graphs Row
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="graph-network", style={"height": "50vh"}))), width=7),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="chart-latency", style={"height": "50vh"}))), width=5),
            ]),
            
            # Second Row of Charts
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="chart-energy", style={"height": "35vh"}))), width=6),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="chart-pdr", style={"height": "35vh"}))), width=6),
            ]),
            
            # P-QCEA Specific Charts
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="chart-prediction-usage", style={"height": "35vh"}))), width=6),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="chart-prediction-confidence", style={"height": "35vh"}))), width=6),
            ]),
            
             # Log
             dbc.Row([
                 dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Simulation Log"),
                    dcc.Textarea(id="sim-log", style={'width': '100%', 'height': '20vh'}, readOnly=True, value="Log messages will appear here..."),
                ])), width=12)
             ])
            
        ], fluid=True, style={'padding': '0 2rem 2rem 2rem'})
    ])