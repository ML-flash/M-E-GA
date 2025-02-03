# experiment_runner_with_dashboard.py
import random
import datetime
from threading import Thread

# -------------------------------
# GA and Fitness Function Imports
# -------------------------------
# Adjust these imports based on your project structure.
from M_E_GA_Base import M_E_GA_Base
from M_E_GA_fitness_funcs import LeadingOnesFitness

# -------------------------------
# Global Settings and Seed
# -------------------------------
MAX_LENGTH = 50000
GLOBAL_SEED = None
random.seed(GLOBAL_SEED)

# -------------------------------
# Best Organism Callback
# -------------------------------
best_organism = {
    "genome": None,
    "fitness": float('-inf')
}

def update_best_organism(current_genome, current_fitness, verbose=False):
    global best_organism
    if current_fitness > best_organism["fitness"]:
        best_organism["genome"] = current_genome
        best_organism["fitness"] = current_fitness
        if verbose:
            print(f"New best organism found with fitness {current_fitness}")

# -------------------------------
# Initialize Fitness Function & Genes
# -------------------------------
fitness_function = LeadingOnesFitness(max_length=MAX_LENGTH, update_best_func=update_best_organism)
genes = fitness_function.genes

# -------------------------------
# GA Configuration Parameters
# -------------------------------
config = {
    'mutation_prob': 0.15,
    'delimited_mutation_prob': 0.05,
    'open_mutation_prob': 0.08,
    'metagene_mutation_prob': 0.09,
    'delimiter_insert_prob': 0.03,
    'delimit_delete_prob': 0.01,
    'crossover_prob': 0.0,
    'elitism_ratio': 0.7,
    'base_gene_prob': 0.44,
    'metagene_prob': 0.04,
    'max_individual_length': 90,
    'population_size': 700,
    'num_parents': 200,
    'max_generations': 200,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': True,
    'generation_logging': True,
    'mutation_logging': True,
    'crossover_logging': True,
    'individual_logging': True,
    'seed': GLOBAL_SEED,
    'lru_cache_size': 50
}

# -------------------------------
# Initialize the GA
# -------------------------------
ga = M_E_GA_Base(genes,
                 lambda ind, ga_instance: fitness_function.compute(ind, ga_instance),
                 **config)

# -------------------------------
# DASH Dashboard Setup
# -------------------------------
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx

app = dash.Dash(__name__)
app.title = "Metagenome Real-Time Dashboard"

app.layout = html.Div([
    html.H1("Real-Time Metagenome Visualization"),
    html.Div(id='stats-div'),
    dcc.Graph(id='dag-graph'),
    # Show recent logger events (if any)
    html.Div(id='logger-events'),
    # Update every 5 seconds
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])


# -------------------------------
# Helper functions for hierarchy computation
# -------------------------------
def compute_metagene_order(mg, encoding_manager, memo):
    """
    Recursively compute the order (hierarchy level) of a metagene.
    Base genes are order 0.
    A metagene that references only base genes gets order 1.
    Otherwise, its order is max(order(child)) + 1.
    """
    # Use memoization to avoid cycles/repeat work.
    if mg in memo:
        return memo[mg]

    # Get the encoding (tuple) for this metagene.
    encoding = encoding_manager.encodings.get(mg, ())
    orders = []
    for element in encoding:
        # If the element is itself a metagene, compute its order.
        if element in encoding_manager.meta_genes:
            child_order = compute_metagene_order(element, encoding_manager, memo)
            orders.append(child_order)
        else:
            # Base genes (uploaded genes) are order 0.
            orders.append(0)
    order = max(orders, default=0) + 1  # at least order 1 if it contains only base genes
    memo[mg] = order
    return order

def get_node_order(node, encoding_manager):
    """
    Return the order for the given node.
    Base genes (uploaded) are order 0.
    For metagenes, compute recursively.
    """
    if node in encoding_manager.meta_genes:
        return compute_metagene_order(node, encoding_manager, memo={})
    else:
        return 0  # base gene

# -------------------------------
# Build the DAG with a Custom Hierarchical Layout
# -------------------------------
def build_dag_custom(encoding_manager):
    """
    Build a directed graph where:
      - Base genes (uploaded genes) are considered the root (order 0).
      - Each captured metagene is added and assigned an order based on its composition.
      - An edge is added from a metagene to each gene it references.
      - The x-axis position is assigned based on the order (e.g. order * constant).
    """
    G = nx.DiGraph()

    # Add base gene nodes.
    # Base genes are those present in the reverse encoding dictionary but not in meta_genes.
    for gene, hash_key in encoding_manager.reverse_encodings.items():
        # We exclude the reserved names 'Start' and 'End'
        if gene in ['Start', 'End']:
            continue
        if hash_key not in encoding_manager.meta_genes:
            G.add_node(hash_key, group="base", label=f"Base: {gene}")

    # Add metagene nodes.
    for mg in encoding_manager.meta_genes:
        # We tag metagenes simply as "meta"
        order = compute_metagene_order(mg, encoding_manager, memo={})
        G.add_node(mg, group="meta", label=f"MG {mg}\n(order {order})", order=order)

    # Add edges: for each metagene, add an edge to each element in its encoding.
    for mg in encoding_manager.meta_genes:
        encoding = encoding_manager.encodings.get(mg)
        if isinstance(encoding, tuple):
            for element in encoding:
                # If the element isn’t already in the graph (e.g. base gene), add it.
                if not G.has_node(element):
                    # It is a base gene.
                    gene_label = encoding_manager.encodings.get(element, element)
                    G.add_node(element, group="base", label=f"Base: {gene_label}")
                G.add_edge(mg, element)

    # Manually assign positions:
    # x coordinate will be proportional to the order.
    # y coordinate: distribute nodes in each order level vertically.
    pos = {}
    # Gather nodes by order.
    order_groups = {}
    for node, data in G.nodes(data=True):
        # For base genes, we assign order 0.
        order_val = data.get("order", 0)
        order_groups.setdefault(order_val, []).append(node)

    # Define spacing constants.
    x_spacing = 0.3
    y_top = 1.0
    y_bottom = 0.0

    for order_val, nodes in order_groups.items():
        x_val = order_val * x_spacing  # e.g., 0 for order 0, 0.3 for order 1, etc.
        nodes.sort(key=lambda n: str(n))
        count = len(nodes)
        for i, node in enumerate(nodes):
            # Evenly space vertically.
            if count > 1:
                y_val = y_top - i * ((y_top - y_bottom) / (count - 1))
            else:
                y_val = 0.5
            pos[node] = (x_val, y_val)

    return G, pos

# -------------------------------
# Dashboard Callbacks
# -------------------------------
@app.callback(Output('stats-div', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_stats(n):
    status = ga.encoding_manager.get_metagene_status()
    return html.Div([
        html.P(f"Current Generation: {status.get('generation', 'N/A')}"),
        html.P(f"Total Metagenes: {status.get('total_metagenes', 0)}"),
        html.P(f"Deletion Basket: {status.get('in_basket', 0)}")
    ])

@app.callback(Output('logger-events', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_logger_events(n):
    # Display the last 5 events from the logger.
    if ga.logger is not None and hasattr(ga.logger, "events"):
        events = ga.logger.events[-5:]
        items = [html.Li(f"{e['timestamp']} - {e['event_type']}: {e['details']}") for e in events]
        return html.Div([html.H4("Recent Logger Events"), html.Ul(items)])
    return "No logger events."

@app.callback(Output('dag-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_dag(n):
    G, pos = build_dag_custom(ga.encoding_manager)

    # Build edge traces.
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Build node traces.
    node_x, node_y, node_text, node_color = [], [], [], []
    # Define colors for groups.
    group_color = {"base": "orange", "meta": "blue"}
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data.get("label", str(node)))
        group = data.get("group", "meta")
        node_color.append(group_color.get(group, "blue"))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            size=20,
            color=node_color,
            line=dict(width=2)
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Metagenome Hierarchy at Generation {ga.encoding_manager.current_generation}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig

def run_dashboard():
    app.run_server(debug=False, port=8050)

# -------------------------------
# Start the Dashboard in a Background Thread
# -------------------------------
dashboard_thread = Thread(target=run_dashboard, daemon=True)
dashboard_thread.start()

# -------------------------------
# Run the GA Algorithm
# -------------------------------
ga.run_algorithm()

# -------------------------------
# After the GA completes, print the best solution
# -------------------------------
best_genome = best_organism["genome"]
best_fitness = best_organism["fitness"]
best_solution_decoded = ga.decode_organism(best_genome, format=True)

print('Length of best solution:', len(best_solution_decoded))
print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
print('Length of best genome:', len(best_genome))
print(f"Best Genome (Encoded): {best_genome}")
