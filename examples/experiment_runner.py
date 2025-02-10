# experiment_runner_with_realtime_integration.py
import random
import threading
import time
import networkx as nx
from threading import Thread
import matplotlib.pyplot as plt
import queue  # For thread-safe communication

# -------------------------------
# GA and Fitness Function Imports
# -------------------------------
from M_E_GA import M_E_GA_Base
from M_E_GA_fitness_funcs import LeadingOnesFitness

# -------------------------------
# Global Settings and Seed
# -------------------------------
MAX_LENGTH = 4000
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
    'mutation_prob': 0.10,
    'delimited_mutation_prob': 0.05,
    'open_mutation_prob': 0.10,
    'metagene_mutation_prob': 0.07,
    'delimiter_insert_prob': 0.03,
    'delimit_delete_prob': 0.02,
    'crossover_prob': 0.0,
    'elitism_ratio': 0.7,
    'base_gene_prob': 0.30,
    'metagene_prob': 0.02,
    'max_individual_length': 50,
    'population_size': 700,
    'num_parents': 300,
    'max_generations': 500,
    'delimiters': False,
    'delimiter_space': 2,
    'logging': True,
    'generation_logging': True,
    'mutation_logging': True,
    'crossover_logging': True,
    'individual_logging': True,
    'seed': GLOBAL_SEED,
    'lru_cache_size': 75
}

# -------------------------------
# Initialize the GA
# -------------------------------
ga = M_E_GA_Base(
    genes,
    lambda ind, ga_instance: fitness_function.compute(ind, ga_instance),
    **config
)
# (If experiment_name is not provided, you'll be prompted.)

# -------------------------------
# Helper Functions for DAG Building
# -------------------------------
def compute_metagene_order(mg, encoding_manager, memo):
    if mg in memo:
        return memo[mg]
    encoding = encoding_manager.encodings.get(mg, ())
    orders = []
    for element in encoding:
        if element in encoding_manager.meta_genes:
            child_order = compute_metagene_order(element, encoding_manager, memo)
            orders.append(child_order)
        else:
            orders.append(0)
    order = max(orders, default=0) + 1
    memo[mg] = order
    return order

def build_dag_custom(encoding_manager):
    G = nx.DiGraph()
    # Add base gene nodes.
    for gene, hash_key in encoding_manager.reverse_encodings.items():
        if gene in ['Start', 'End']:
            continue
        if hash_key not in encoding_manager.meta_genes:
            G.add_node(hash_key, group="base", label=f"Base: {gene}")
    # Add metagene nodes.
    for mg in encoding_manager.meta_genes:
        order = compute_metagene_order(mg, encoding_manager, memo={})
        G.add_node(mg, group="meta", label=f"MG {mg}\n(order {order})", order=order)
    # Add edges from each metagene to the genes it references.
    for mg in encoding_manager.meta_genes:
        encoding = encoding_manager.encodings.get(mg)
        if isinstance(encoding, tuple):
            for element in encoding:
                if not G.has_node(element):
                    gene_label = encoding_manager.encodings.get(element, element)
                    G.add_node(element, group="base", label=f"Base: {gene_label}")
                G.add_edge(mg, element)
    # Create a custom layout.
    pos = {}
    order_groups = {}
    for node, data in G.nodes(data=True):
        order_val = data.get("order", 0)
        order_groups.setdefault(order_val, []).append(node)
    x_spacing = 0.3
    y_top = 1.0
    y_bottom = 0.0
    for order_val, nodes in order_groups.items():
        x_val = order_val * x_spacing
        nodes.sort(key=lambda n: str(n))
        count = len(nodes)
        for i, node in enumerate(nodes):
            y_val = y_top - i * ((y_top - y_bottom) / (count - 1)) if count > 1 else 0.5
            pos[node] = (x_val, y_val)
    return G, pos

# -------------------------------
# Flask and SocketIO Setup
# -------------------------------
from flask import Flask, jsonify
from flask_socketio import SocketIO

app = Flask(__name__, static_folder='../react_app/build', static_url_path='/')
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/api/stats')
def stats():
    status = ga.encoding_manager.get_metagene_status()
    return jsonify({
        'generation': status.get('generation', 'N/A'),
        'total_metagenes': status.get('total_metagenes', 0),
        'in_basket': status.get('in_basket', 0)
    })

@app.route('/api/dag')
def dag():
    G, pos = build_dag_custom(ga.encoding_manager)
    nodes = []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        nodes.append({
            'id': node,
            'group': data.get("group", "meta"),
            'label': data.get("label", str(node)),
            'x': x,
            'y': y
        })
    edges = []
    for source, target in G.edges():
        edges.append({'source': source, 'target': target})
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'generation': ga.encoding_manager.current_generation
    })

@app.route('/')
def index():
    return app.send_static_file('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected via SocketIO')

def background_thread():
    """Continuously emit GA updates over SocketIO."""
    while True:
        socketio.sleep(1)
        G, pos = build_dag_custom(ga.encoding_manager)
        nodes = []
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            nodes.append({
                'id': node,
                'group': data.get("group", "meta"),
                'label': data.get("label", str(node)),
                'x': x,
                'y': y
            })
        edges = []
        for source, target in G.edges():
            edges.append({'source': source, 'target': target})
        payload = {
            'nodes': nodes,
            'edges': edges,
            'generation': ga.encoding_manager.current_generation
        }
        socketio.emit('dag_update', payload)

socketio.start_background_task(target=background_thread)

# -------------------------------
# Native Real-Time Plotting Setup Using a Queue
# -------------------------------
# Create a thread-safe queue to signal when a plot update is requested.
update_queue = queue.Queue()

def update_plot_callback(event):
    """
    This callback is invoked by the GA logger.
    Instead of updating the plot directly from a background thread,
    we push a message into the update queue.
    """
    if event['event_type'] in ("generation_summary", "metagene_captured", "metagene_deleted"):
        print(f"[Plot] Event: {event['event_type']} at {event['timestamp']}")
        update_queue.put("update")

if ga.logger:
    ga.logger.subscribe(update_plot_callback)

def update_plot():
    """Perform the actual matplotlib plot update (this runs on the main thread)."""
    G, pos = build_dag_custom(ga.encoding_manager)
    plt.clf()
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=600, font_size=8)
    plt.title(f"Generation: {ga.encoding_manager.current_generation}")
    plt.draw()

# -------------------------------
# GA Run Function
# -------------------------------
def run_ga():
    ga.run_algorithm()
    best_genome = best_organism["genome"]
    best_fitness = best_organism["fitness"]
    best_solution_decoded = ga.decode_organism(best_genome, format=True)
    print('GA completed')
    print('Length of best solution:', len(best_solution_decoded))
    print(f"Best Solution (Decoded): {best_solution_decoded}, Fitness: {best_fitness}")
    print('Length of best genome:', len(best_genome))
    print(f"Best Genome (Encoded): {best_genome}")

# -------------------------------
# Main Section: Start Threads and Run Main GUI Loop
# -------------------------------
if __name__ == '__main__':
    # Start the GA in its own thread.
    ga_thread = Thread(target=run_ga)
    ga_thread.daemon = True
    ga_thread.start()

    # Start the SocketIO server in its own thread.
    def run_socketio():
        socketio.run(app, debug=False, port=5000)
    socketio_thread = Thread(target=run_socketio)
    socketio_thread.daemon = True
    socketio_thread.start()

    # Set up matplotlib in interactive mode and force the window to appear.
    plt.ion()
    fig = plt.figure()
    plt.show(block=False)

    print("Starting main GUI loop in the main thread...")
    try:
        while True:
            # Process all pending update messages.
            while not update_queue.empty():
                try:
                    msg = update_queue.get_nowait()
                    if msg == "update":
                        update_plot()
                except queue.Empty:
                    break
            # Always call plt.pause to allow the GUI event loop to process.
            plt.pause(0.1)
    except KeyboardInterrupt:
        print("Exiting main GUI loop.")
