import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import random
import yaml

# --- Import from layout file ---
from .layout import (
    create_network_figure, 
    create_empty_figure, 
    create_kpi_card,
    PQCEA_COLOR,
    QCEA_COLOR,
    DIJKSTRA_COLOR,
    PLOT_BG_COLOR,
    PLOT_FONT_COLOR,
    GRID_COLOR
)

# --- !!! IMPORT YOUR ACTUAL SRC CODE !!! ---
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.pqcea_routing import PQCEARouting
    from src.qcea_routing import QCEARouting
    from src.predictor import LinkPredictor
    from src.baseline_dijkstra import run_dijkstra
    from src.topology import (
        create_scale_free_topology, 
        simple_dynamic_update, 
        deduct_path_energy,
        load_graph_yaml
    )
    from src.traffic import generate_flows
    print("✅ Successfully imported all src modules")
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print("Falling back to mock functions. Please check 'src' imports and file names.")
    # Fallback mock functions
    class MockRouter:
        def __init__(self, *args, **kwargs): pass
        def compute_path(self, G, src, dst, **kwargs):
            return nx.shortest_path(G, src, dst), 10, {}
        def update_link_measurements(self, G): pass
        def get_statistics(self): return {'prediction_usage_rate': 0.5, 'predictor': {'avg_confidence': 0.8}}
    PQCEARouting = MockRouter
    QCEARouting = MockRouter
    class MockPredictor:
        def __init__(self, *args, **kwargs): pass
    LinkPredictor = MockPredictor
    def run_dijkstra(G, source, target, **kwargs):
        return nx.shortest_path(G, source, target), 10
    def create_scale_free_topology(*args, **kwargs): return nx.path_graph(10)
    def simple_dynamic_update(*args, **kwargs): pass
    def deduct_path_energy(*args, **kwargs): return 10
    def generate_flows(*args, **kwargs):
        G = kwargs.get('G'); nodes = list(G.nodes()); return [{"src": nodes[0], "dst": nodes[-1]}]
    def load_graph_yaml(*args, **kwargs):
        G = nx.barabasi_albert_graph(n=20, m=2, seed=42)
        for node in G.nodes():
            G.nodes[node]['residual_energy_j'] = 10000.0
            G.nodes[node]['role'] = 'edge' if G.degree(node) < 3 else 'core'
        for u, v in G.edges():
            G[u][v]['prop_delay_ms'] = random.uniform(5, 20)
            G[u][v]['bandwidth_bps'] = random.choice([10e6, 50e6, 100e6])
            G[u][v]['loss_prob'] = random.uniform(0.0, 0.05)
        return G
        
# --- Core Simulation Step Function ---

def run_simulation_step(G, t, pqcea_router, qcea_router, num_flows=5, traffic_mix=None, weight_mode='auto'):
    """
    MODIFIED: Runs ONE step for ALL THREE algorithms with traffic mix support.
    Supports AUTO mode (per-traffic weights) and MANUAL mode (global weights).
    """
    # 1. Update network state
    simple_dynamic_update(G, t)
    
    # 2. Update P-QCEA predictor
    pqcea_router.update_link_measurements(G)
    
    # 3. Reset link utilization for this step
    for u, v, data in G.edges(data=True):
        data['utilization'] = 0.0
        data['active_flows'] = 0
    
    # 4. Generate flows with traffic mix
    flows = generate_flows(G, num_flows=num_flows, seed=t, traffic_mix=traffic_mix) 
    
    metrics_pqcea = {'latency': [], 'throughput': [], 'energy': 0, 'delivered': 0, 'jitter': []}
    metrics_qcea = {'latency': [], 'throughput': [], 'energy': 0, 'delivered': 0, 'jitter': []}
    metrics_dijkstra = {'latency': [], 'throughput': [], 'energy': 0, 'delivered': 0, 'jitter': []}
    paths = {'pqcea': [], 'qcea': [], 'dijkstra': []}
    log_msgs = []
    
    pkt_size = 8 * 1000  # bits
    
    # --- Graph copy for Dijkstra (to add 'weight' attribute) ---
    G_dijkstra = G.copy() 
    for u, v, data in G_dijkstra.edges(data=True):
        data['weight'] = data.get('prop_delay_ms', 1.0)
        
    # --- *** FIX 1: Create a DEEP COPY of the graph for energy calculation *** ---
    # This is critical. We need to deduct energy from separate graphs.
    G_pqcea_energy = G.copy()
    G_qcea_energy = G.copy()
    G_dijkstra_energy = G.copy()
    # -------------------------------------------------------------------------
    
    for flow in flows:
        src, dst = flow["src"], flow["dst"]
        traffic_class = flow.get("class", "besteffort")
        
        # Apply weights based on mode
        if weight_mode == 'auto':
            # AUTO MODE: Use traffic-specific weights per flow
            from src.traffic import get_traffic_weights
            flow_weights = get_traffic_weights(traffic_class)
            pqcea_router.weights = flow_weights
            qcea_router.weights = flow_weights
        # else: MANUAL MODE: Keep the manual_weights already set
        
        # --- 1. P-QCEA ALGORITHM (Predictive) ---
        try:
            path, cost, info = pqcea_router.compute_path(G, src, dst, mode='predictive') 
            if path:
                paths['pqcea'].append(path)
                latency = sum(G.edges[u, v]['prop_delay_ms'] for u, v in zip(path[:-1], path[1:]))
                metrics_pqcea['latency'].append(latency)
                metrics_pqcea['throughput'].append(1.0 / (latency + 1e-6)) 
                
                # Update link utilization
                for u, v in zip(path[:-1], path[1:]):
                    G[u][v]['active_flows'] += 1
                    # Simple utilization: flows / bandwidth (normalized)
                    G[u][v]['utilization'] = min(1.0, G[u][v]['active_flows'] * 0.1)
                
                # Deduct energy from its OWN graph copy
                energy = deduct_path_energy(G_pqcea_energy, path, pkt_size)
                metrics_pqcea['energy'] += energy
                
                # --- *** FIX 2: Correct PDR Calculation *** ---
                survival_prob = 1.0
                for u, v in zip(path[:-1], path[1:]):
                    survival_prob *= (1.0 - G.edges[u, v]['loss_prob'])
                if random.random() < survival_prob:
                # -----------------------------------------------
                    metrics_pqcea['delivered'] += 1
            else:
                 log_msgs.append(f"[P-QCEA] No path {src}-{dst}")
        except Exception as e:
            log_msgs.append(f"[P-QCEA] Error {src}-{dst}: {e}")

        # --- 2. QCEA ALGORITHM (Reactive) ---
        try:
            path_q, cost_q = qcea_router.compute_path(G, src, dst) 
            if path_q:
                paths['qcea'].append(path_q)
                latency_q = sum(G.edges[u, v]['prop_delay_ms'] for u, v in zip(path_q[:-1], path_q[1:]))
                metrics_qcea['latency'].append(latency_q)
                metrics_qcea['throughput'].append(1.0 / (latency_q + 1e-6))
                
                # Update link utilization
                for u, v in zip(path_q[:-1], path_q[1:]):
                    G[u][v]['active_flows'] += 1
                    G[u][v]['utilization'] = min(1.0, G[u][v]['active_flows'] * 0.1)
                
                # Deduct energy from its OWN graph copy
                energy_q = deduct_path_energy(G_qcea_energy, path_q, pkt_size) 
                metrics_qcea['energy'] += energy_q
                
                # --- *** FIX 2: Correct PDR Calculation *** ---
                survival_prob_q = 1.0
                for u, v in zip(path_q[:-1], path_q[1:]):
                    survival_prob_q *= (1.0 - G.edges[u, v]['loss_prob'])
                if random.random() < survival_prob_q:
                # -----------------------------------------------
                    metrics_qcea['delivered'] += 1
            else:
                log_msgs.append(f"[QCEA] No path {src}-{dst}")
        except Exception as e:
            log_msgs.append(f"[QCEA] Error {src}-{dst}: {e}")

        # --- 3. DIJKSTRA ALGORITHM (Baseline) ---
        try:
            path_d, cost_d = run_dijkstra(G_dijkstra, src, dst) 
            if path_d:
                paths['dijkstra'].append(path_d)
                metrics_dijkstra['latency'].append(cost_d) 
                metrics_dijkstra['throughput'].append(1.0 / (cost_d + 1e-6))
                
                # Deduct energy from its OWN graph copy
                energy_d = deduct_path_energy(G_dijkstra_energy, path_d, pkt_size) 
                metrics_dijkstra['energy'] += energy_d
                
                # --- *** FIX 2: Correct PDR Calculation *** ---
                survival_prob_d = 1.0
                for u, v in zip(path_d[:-1], path_d[1:]):
                    survival_prob_d *= (1.0 - G.edges[u, v]['loss_prob'])
                if random.random() < survival_prob_d:
                # -----------------------------------------------
                    metrics_dijkstra['delivered'] += 1
            else:
                log_msgs.append(f"[DIJK] No path {src}-{dst}")
        except Exception as e:
            log_msgs.append(f"[DIJK] Error {src}-{dst}: {e}")

    # Calculate jitter (standard deviation of latency)
    if len(metrics_pqcea['latency']) > 1:
        metrics_pqcea['jitter'] = np.std(metrics_pqcea['latency'])
    else:
        metrics_pqcea['jitter'] = 0.0
    
    if len(metrics_qcea['latency']) > 1:
        metrics_qcea['jitter'] = np.std(metrics_qcea['latency'])
    else:
        metrics_qcea['jitter'] = 0.0
    
    if len(metrics_dijkstra['latency']) > 1:
        metrics_dijkstra['jitter'] = np.std(metrics_dijkstra['latency'])
    else:
        metrics_dijkstra['jitter'] = 0.0
    
    # --- Aggregate metrics ---
    def aggregate(metrics, num_f):
        return {
            'latency': np.mean(metrics['latency']) if metrics['latency'] else 0,
            'throughput': np.mean(metrics['throughput']) * 1e3 if metrics['throughput'] else 0, # -> Mbps
            'pdr': (metrics['delivered'] / num_f) * 100.0 if num_f > 0 else 0,
            'energy_consumed': metrics['energy'], # Report CONSUMED energy
            'jitter': metrics.get('jitter', 0.0),
        }
    
    num_flows_actual = len(flows)
    aggr_pqcea = aggregate(metrics_pqcea, num_flows_actual)
    aggr_qcea = aggregate(metrics_qcea, num_flows_actual)
    aggr_dijkstra = aggregate(metrics_dijkstra, num_flows_actual)
    
    # --- *** FIX 1 (cont.): Report RESIDUAL energy from the *correct* graph *** ---
    aggr_pqcea['residual_energy'] = sum(d['residual_energy_j'] for _, d in G_pqcea_energy.nodes(data=True))
    aggr_qcea['residual_energy'] = sum(d['residual_energy_j'] for _, d in G_qcea_energy.nodes(data=True))
    aggr_dijkstra['residual_energy'] = sum(d['residual_energy_j'] for _, d in G_dijkstra_energy.nodes(data=True))
    # -------------------------------------------------------------------------

    # Get prediction stats for this step
    # Calculate path-level usage rate (not link-level)
    num_paths_with_predictions = 0
    for path in paths['pqcea']:
        if len(path) > 1:
            # Check if first link in path used prediction
            u, v = path[0], path[1]
            pred_result = pqcea_router.predictor.predict(u, v, steps_ahead=1)
            if pred_result and pred_result.get('use_prediction', False):
                num_paths_with_predictions += 1
    
    usage_rate = (num_paths_with_predictions / len(paths['pqcea']) * 100) if paths['pqcea'] else 0
    
    # Debug: Print stats
    pred_stats = pqcea_router.get_statistics()
    predictor_stats = pred_stats.get('predictor', {})
    num_links_tracked = predictor_stats.get('num_links_tracked', 0)
    avg_history = predictor_stats.get('avg_history_size', 0)
    print(f"[DEBUG Step {t}] Prediction: paths={len(paths['pqcea'])}, " +
          f"with_pred={num_paths_with_predictions}, " +
          f"tracked={num_links_tracked}, history={avg_history:.1f}, usage={usage_rate:.0f}%")
    
    # Calculate average confidence from predictions made this step
    avg_conf = 0
    try:
        confidences = []
        # Check all links that were used in paths this step
        for path in paths['pqcea']:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                pred_result = pqcea_router.predictor.predict(u, v, steps_ahead=1)
                if pred_result and 'confidence' in pred_result:
                    conf_dict = pred_result['confidence']
                    # Average of delay, bandwidth, loss confidences
                    avg_link_conf = np.mean([
                        conf_dict.get('delay', 0),
                        conf_dict.get('bandwidth', 0),
                        conf_dict.get('loss', 0)
                    ])
                    confidences.append(avg_link_conf)
        avg_conf = np.mean(confidences) if confidences else 0
    except Exception as e:
        # Fallback: use overall confidence if available
        try:
            if len(pqcea_router.predictor.link_history) > 0:
                # Sample confidence from first few tracked links
                sample_confs = []
                for link_key in list(pqcea_router.predictor.link_history.keys())[:5]:
                    u, v = link_key
                    pred = pqcea_router.predictor.predict(u, v, steps_ahead=1)
                    if pred and 'confidence' in pred:
                        sample_confs.append(pred['confidence'].get('average', 0))
                avg_conf = np.mean(sample_confs) if sample_confs else 0.5
            else:
                avg_conf = 0.5  # Default moderate confidence
        except:
            avg_conf = 0.5
    
    aggr_pqcea['prediction_usage_rate'] = usage_rate
    aggr_pqcea['avg_confidence'] = avg_conf * 100 # as % 

    logs = f"[P] Lat: {aggr_pqcea['latency']:.1f}ms, PDR: {aggr_pqcea['pdr']:.0f}% | [Q] Lat: {aggr_qcea['latency']:.1f}ms | [D] Lat: {aggr_dijkstra['latency']:.1f}ms"
    
    return G, aggr_pqcea, aggr_qcea, aggr_dijkstra, paths, logs


# --- Callback Registration ---

def register_callbacks(app):
    
    @app.callback(
        Output("collapse-controls", "is_open"),
        [Input("btn-toggle-controls", "n_clicks")],
        [State("collapse-controls", "is_open")],
    )
    def toggle_controls_collapse(n, is_open):
        if n: return not is_open
        return is_open

    @app.callback(
        Output('qcea-weights-panel', 'style'),
        Input('radio-algo', 'value')
    )
    def toggle_qcea_sliders(algo):
        if algo in ['qcea', 'pqcea']:
            return {'display': 'block'}
        return {'display': 'none'}

    @app.callback(
        Output('manual-weight-sliders', 'style'),
        Input('radio-weight-mode', 'value')
    )
    def toggle_manual_sliders(mode):
        if mode == 'manual':
            return {'display': 'block', 'marginTop': '1rem'}
        return {'display': 'none'}

    @app.callback(
        Output('prediction-params-panel', 'style'),
        Input('radio-algo', 'value')
    )
    def toggle_prediction_sliders(algo):
        if algo == 'pqcea':
            return {'display': 'block'}
        return {'display': 'none'}

    @app.callback(
        [Output('store-sim-state', 'data'),
         Output('interval-sim-step', 'disabled'),
         Output('sim-status', 'children'),
         Output('btn-start', 'disabled'),
         Output('btn-stop', 'disabled'),
         Output('store-sim-data-dijkstra', 'data'),
         Output('store-sim-data-qcea', 'data'),
         Output('store-sim-data-pqcea', 'data')],
        [Input('btn-start', 'n_clicks'),
         Input('btn-stop', 'n_clicks'),
         Input('interval-sim-step', 'n_intervals')],
        [State('store-sim-state', 'data'),
         State('slider-time-steps', 'value'),
         State('slider-nodes', 'value'),
         State('input-seed', 'value')] 
    )
    def manage_simulation_state(start_clicks, stop_clicks, n_intervals, sim_state, max_steps, num_nodes, seed):
        """Manages the main simulation loop (Start, Stop, Step)."""
        ctx = callback_context
        if not ctx.triggered: raise dash.exceptions.PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'btn-start':
            print("--- SIMULATION START ---")
            # Set random seed for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            print(f"Random seed set to: {seed}")
            
            # Create scale-free topology with user-specified number of nodes
            G = create_scale_free_topology(n=num_nodes, m=2, seed=seed)
            print(f"Created scale-free topology with {num_nodes} nodes")
            
            G_json = nx.node_link_data(G)
            new_state = {'running': True, 'step': 0, 'max_steps': max_steps}
            store_dijkstra = {'G_json': G_json, 'metrics': []}
            store_qcea = {'G_json': G_json, 'metrics': []}
            store_pqcea = {'G_json': G_json, 'metrics': [], 'predictor_history': {}}
            return new_state, False, "Status: Running (Step 0)", True, False, store_dijkstra, store_qcea, store_pqcea
        
        if trigger_id == 'btn-stop':
            print("--- SIMULATION STOP ---")
            new_state = {'running': False, 'step': sim_state['step'], 'max_steps': sim_state['max_steps']}
            return new_state, True, f"Status: Stopped (Step {sim_state['step']})", False, True, dash.no_update, dash.no_update, dash.no_update
        
        if trigger_id == 'interval-sim-step':
            if not sim_state['running']: raise dash.exceptions.PreventUpdate
            current_step = sim_state['step']
            if current_step >= sim_state['max_steps']:
                new_state = {'running': False, 'step': current_step, 'max_steps': sim_state['max_steps']}
                return new_state, True, f"Status: Complete (Step {current_step})", False, True, dash.no_update, dash.no_update, dash.no_update
            new_step = current_step + 1
            new_state = {'running': True, 'step': new_step, 'max_steps': sim_state['max_steps']}
            status_msg = f"Status: Running (Step {new_step} / {sim_state['max_steps']})"
            return new_state, False, status_msg, True, False, dash.no_update, dash.no_update, dash.no_update
        
        raise dash.exceptions.PreventUpdate

    @app.callback(
        [Output('graph-network', 'figure'),
         Output('store-sim-data-dijkstra', 'data', allow_duplicate=True),
         Output('store-sim-data-qcea', 'data', allow_duplicate=True),
         Output('store-sim-data-pqcea', 'data', allow_duplicate=True),
         Output('sim-log', 'value')],
        [Input('store-sim-state', 'data')], 
        [State('store-sim-data-dijkstra', 'data'),
         State('store-sim-data-qcea', 'data'),
         State('store-sim-data-pqcea', 'data'),
         State('radio-algo', 'value'),
         State('input-flows', 'value'),
         State('sim-log', 'value'),
         State('slider-wl', 'value'), 
         State('slider-wb', 'value'),
         State('slider-wp', 'value'), 
         State('slider-we', 'value'),
         State('slider-wc', 'value'),
         State('slider-horizon', 'value'),
         State('slider-alpha', 'value'),
         State('slider-confidence', 'value'),
         State('slider-voip-mix', 'value'),
         State('slider-video-mix', 'value'),
         State('slider-data-mix', 'value'),
         State('radio-weight-mode', 'value')],
        prevent_initial_call=True
    )
    def run_simulation_step_callback(sim_state, dijkstra_data, qcea_data, pqcea_data, 
                                     selected_algo, num_flows, log_text, 
                                     wl, wb, wp, we, wc,
                                     horizon, alpha, confidence,
                                     voip_mix, video_mix, data_mix, weight_mode):
        """
        MODIFIED: This is the core logic. It runs all 3 algorithms.
        Supports both AUTO (per-traffic weights) and MANUAL (global weights) modes.
        """
        if not sim_state['running'] or sim_state['step'] == 0:
            raise dash.exceptions.PreventUpdate
        
        # Initialize routers
        # In AUTO mode: these are overridden per-flow
        # In MANUAL mode: these are used globally
        manual_weights = {"wl": wl, "wb": wb, "wp": wp, "we": we, "wc": wc}
        qcea_router = QCEARouting(manual_weights)
        
        pqcea_router = PQCEARouting(manual_weights, prediction_horizon=horizon)
        if hasattr(pqcea_router, 'predictor'):
            pqcea_router.predictor.alpha = alpha
            pqcea_router.predictor.confidence_threshold = confidence
            
            # Restore predictor history from previous steps (convert lists back to deques)
            if pqcea_data.get('predictor_history'):
                from collections import deque
                hist_data = pqcea_data['predictor_history']
                history_size = hist_data.get('history_size', 15)
                
                for key_str, hist in hist_data.get('link_history', {}).items():
                    key = eval(key_str)  # Convert string back to tuple
                    pqcea_router.predictor.link_history[key] = {
                        'delay': deque(hist['delay'], maxlen=history_size),
                        'bandwidth': deque(hist['bandwidth'], maxlen=history_size),
                        'loss': deque(hist['loss'], maxlen=history_size)
                    }
                
                for key_str, val in hist_data.get('smoothed_values', {}).items():
                    key = eval(key_str)
                    pqcea_router.predictor.smoothed_values[key] = val
        
        # We use ONE master graph, store in P-QCEA store
        G = nx.node_link_graph(pqcea_data['G_json'])
        
        # Normalize traffic mix
        total = voip_mix + video_mix + data_mix
        if total == 0:
            total = 1
        traffic_mix = {
            "voip": voip_mix / total,
            "video": video_mix / total,
            "besteffort": data_mix / total
        }
        
        G_new, metrics_p, metrics_q, metrics_d, paths, log_str = run_simulation_step(
            G, sim_state['step'], pqcea_router, qcea_router, num_flows, 
            traffic_mix=traffic_mix, weight_mode=weight_mode
        )
        
        pqcea_data['metrics'].append(metrics_p)
        qcea_data['metrics'].append(metrics_q)
        dijkstra_data['metrics'].append(metrics_d)
        
        # Save predictor history for next step (convert deques to lists for JSON)
        if hasattr(pqcea_router, 'predictor'):
            link_history_serializable = {}
            for key, hist in pqcea_router.predictor.link_history.items():
                link_history_serializable[str(key)] = {
                    'delay': list(hist['delay']),
                    'bandwidth': list(hist['bandwidth']),
                    'loss': list(hist['loss'])
                }
            pqcea_data['predictor_history'] = {
                'link_history': link_history_serializable,
                'smoothed_values': {str(k): v for k, v in pqcea_router.predictor.smoothed_values.items()},
                'history_size': pqcea_router.predictor.history_size
            }
        
        G_json = nx.node_link_data(G_new)
        pqcea_data['G_json'] = G_json
        qcea_data['G_json'] = G_json
        dijkstra_data['G_json'] = G_json
        
        if selected_algo == 'pqcea':
            network_fig = create_network_figure(G_new, paths['pqcea'])
        elif selected_algo == 'qcea':
            network_fig = create_network_figure(G_new, paths['qcea'])
        else:
            network_fig = create_network_figure(G_new, paths['dijkstra'])
            
        new_log = f"[T={sim_state['step']}] {log_str}\n{log_text}"
        if len(new_log) > 2000: new_log = new_log[:2000]
        
        return network_fig, dijkstra_data, qcea_data, pqcea_data, new_log

    @app.callback(
        [Output('chart-latency', 'figure'),
         Output('chart-energy', 'figure'),
         Output('chart-pdr', 'figure'),
         Output('kpi-dijkstra-latency', 'children'),
         Output('kpi-qcea-latency', 'children'),
         Output('kpi-pqcea-latency', 'children'),
         Output('kpi-dijkstra-energy', 'children'),
         Output('kpi-qcea-energy', 'children'),
         Output('kpi-pqcea-energy', 'children'),
         Output('kpi-dijkstra-pdr', 'children'),
         Output('kpi-qcea-pdr', 'children'),
         Output('kpi-pqcea-pdr', 'children'),
         Output('kpi-dijkstra-jitter', 'children'),
         Output('kpi-qcea-jitter', 'children'),
         Output('kpi-pqcea-jitter', 'children'),
         Output('chart-prediction-usage', 'figure'),
         Output('chart-prediction-confidence', 'figure')],
        [Input('store-sim-data-dijkstra', 'data'),
         Input('store-sim-data-qcea', 'data'),
         Input('store-sim-data-pqcea', 'data')],
    )
    def update_kpi_charts(dijkstra_data, qcea_data, pqcea_data):
        """
        MODIFIED: Updates all KPIs and charts for 3 algorithms.
        """
        if not pqcea_data['metrics']:
            return (
                create_empty_figure("Avg. Path Latency"),
                create_empty_figure("Total Residual Network Energy"),
                create_empty_figure("Packet Delivery Rate (PDR)"),
                create_kpi_card("Dijkstra", "N/A", DIJKSTRA_COLOR),
                create_kpi_card("QCEA", "N/A", QCEA_COLOR),
                create_kpi_card("P-QCEA", "N/A", PQCEA_COLOR),
                create_kpi_card("Dijkstra", "N/A", DIJKSTRA_COLOR),
                create_kpi_card("QCEA", "N/A", QCEA_COLOR),
                create_kpi_card("P-QCEA", "N/A", PQCEA_COLOR),
                create_kpi_card("Dijkstra", "N/A", DIJKSTRA_COLOR),
                create_kpi_card("QCEA", "N/A", QCEA_COLOR),
                create_kpi_card("P-QCEA", "N/A", PQCEA_COLOR),
                create_kpi_card("Dijkstra", "N/A", DIJKSTRA_COLOR),
                create_kpi_card("QCEA", "N/A", QCEA_COLOR),
                create_kpi_card("P-QCEA", "N/A", PQCEA_COLOR),
                create_empty_figure("Prediction Usage Rate (%)"),
                create_empty_figure("Avg. Prediction Confidence (%)"),
            )
        
        d_metrics = dijkstra_data['metrics']
        q_metrics = qcea_data['metrics']
        p_metrics = pqcea_data['metrics']
        
        time_steps = list(range(1, len(p_metrics) + 1))
        
        d_latency = [m['latency'] for m in d_metrics]
        q_latency = [m['latency'] for m in q_metrics]
        p_latency = [m['latency'] for m in p_metrics]
        
        # --- *** FIX 1 (cont.): Plot RESIDUAL energy, not consumed *** ---
        d_energy = [m['residual_energy'] for m in d_metrics]
        q_energy = [m['residual_energy'] for m in q_metrics]
        p_energy = [m['residual_energy'] for m in p_metrics]
        # -------------------------------------------------------------

        d_pdr = [m['pdr'] for m in d_metrics]
        q_pdr = [m['pdr'] for m in q_metrics]
        p_pdr = [m['pdr'] for m in p_metrics]
        
        # Jitter data
        d_jitter = [m.get('jitter', 0.0) for m in d_metrics]
        q_jitter = [m.get('jitter', 0.0) for m in q_metrics]
        p_jitter = [m.get('jitter', 0.0) for m in p_metrics]
        
        p_usage = [m['prediction_usage_rate'] for m in p_metrics]
        p_confidence = [m['avg_confidence'] for m in p_metrics]

        # --- Create Chart Figures ---
        def create_line_chart(title, y_title, d_data, q_data, p_data):
            fig = go.Figure(layout=go.Layout(
                title=title,
                xaxis=dict(title='Time Step', gridcolor=GRID_COLOR, zeroline=False),
                yaxis=dict(title=y_title, gridcolor=GRID_COLOR, zeroline=False),
                paper_bgcolor=PLOT_BG_COLOR,
                plot_bgcolor=PLOT_BG_COLOR,
                font_color=PLOT_FONT_COLOR,
                legend=dict(x=0.01, y=0.99, bordercolor=PLOT_FONT_COLOR, borderwidth=1),
                margin=dict(t=40, b=40, l=40, r=20)
            ))
            fig.add_trace(go.Scatter(x=time_steps, y=d_data, mode='lines', name='Dijkstra', line=dict(color=DIJKSTRA_COLOR, width=3)))
            fig.add_trace(go.Scatter(x=time_steps, y=q_data, mode='lines', name='QCEA', line=dict(color=QCEA_COLOR, width=3)))
            fig.add_trace(go.Scatter(x=time_steps, y=p_data, mode='lines', name='P-QCEA', line=dict(color=PQCEA_COLOR, width=3))) 
            return fig
        
        def create_single_line_chart(title, y_title, p_data, color):
            fig = go.Figure(layout=go.Layout(
                title=title,
                xaxis=dict(title='Time Step', gridcolor=GRID_COLOR, zeroline=False),
                yaxis=dict(title=y_title, gridcolor=GRID_COLOR, zeroline=False, range=[0, 100]),
                paper_bgcolor=PLOT_BG_COLOR,
                plot_bgcolor=PLOT_BG_COLOR,
                font_color=PLOT_FONT_COLOR,
                legend=dict(x=0.01, y=0.99, bordercolor=PLOT_FONT_COLOR, borderwidth=1),
                margin=dict(t=40, b=40, l=40, r=20)
            ))
            fig.add_trace(go.Scatter(x=time_steps, y=p_data, mode='lines', name='P-QCEA', line=dict(color=color, width=3)))
            return fig

        chart_latency = create_line_chart("Avg. Path Latency (ms)", "Latency (ms)", d_latency, q_latency, p_latency)
        chart_energy = create_line_chart("Total Residual Network Energy (J)", "Energy (J)", d_energy, q_energy, p_energy)
        chart_pdr = create_line_chart("Packet Delivery Rate (%)", "PDR (%)", d_pdr, q_pdr, p_pdr)
        
        chart_pred_usage = create_single_line_chart("Prediction Usage Rate (%)", "Usage (%)", p_usage, PQCEA_COLOR)
        chart_pred_conf = create_single_line_chart("Avg. Prediction Confidence (%)", "Confidence (%)", p_confidence, QCEA_COLOR)

        # --- Update KPI Cards (4x3 grid) ---
        # Latency
        kpi_d_latency = create_kpi_card("Dijkstra", f"{np.mean(d_latency):.2f} ms", DIJKSTRA_COLOR)
        kpi_q_latency = create_kpi_card("QCEA", f"{np.mean(q_latency):.2f} ms", QCEA_COLOR)
        kpi_p_latency = create_kpi_card("P-QCEA", f"{np.mean(p_latency):.2f} ms", PQCEA_COLOR)
        # Energy (use last value)
        kpi_d_energy = create_kpi_card("Dijkstra", f"{d_energy[-1]:.0f} J", DIJKSTRA_COLOR)
        kpi_q_energy = create_kpi_card("QCEA", f"{q_energy[-1]:.0f} J", QCEA_COLOR)
        kpi_p_energy = create_kpi_card("P-QCEA", f"{p_energy[-1]:.0f} J", PQCEA_COLOR)
        # PDR
        kpi_d_pdr = create_kpi_card("Dijkstra", f"{np.mean(d_pdr):.2f} %", DIJKSTRA_COLOR)
        kpi_q_pdr = create_kpi_card("QCEA", f"{np.mean(q_pdr):.2f} %", QCEA_COLOR)
        kpi_p_pdr = create_kpi_card("P-QCEA", f"{np.mean(p_pdr):.2f} %", PQCEA_COLOR)
        # Jitter (NEW)
        kpi_d_jitter = create_kpi_card("Dijkstra", f"{np.mean(d_jitter):.2f} ms", DIJKSTRA_COLOR)
        kpi_q_jitter = create_kpi_card("QCEA", f"{np.mean(q_jitter):.2f} ms", QCEA_COLOR)
        kpi_p_jitter = create_kpi_card("P-QCEA", f"{np.mean(p_jitter):.2f} ms", PQCEA_COLOR)

        return (
            chart_latency, chart_energy, chart_pdr,
            kpi_d_latency, kpi_q_latency, kpi_p_latency,
            kpi_d_energy, kpi_q_energy, kpi_p_energy,
            kpi_d_pdr, kpi_q_pdr, kpi_p_pdr,
            kpi_d_jitter, kpi_q_jitter, kpi_p_jitter,
            chart_pred_usage, chart_pred_conf
        )