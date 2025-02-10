import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx
import random

# Dummy Data (for demonstration purposes)
meta_genes = {f"MG{i}": {"order": i, "content": [random.randint(1, 100) for _ in range(5)]} for i in range(1, 6)}
edges = [(f"MG{i}", f"Base{j}") for i in range(1, 4) for j in range(1, 3)]

# Create a NetworkX Graph (for demonstration purposes)
G = nx.DiGraph()
G.add_nodes_from([("Start", {"group": "base"}), ("End", {"group": "base"})])
G.add_nodes_from(meta_genes)
G.add_edges_from(edges)
pos = {node: (meta_genes[node]["order"] * 0.3, meta_genes[node]["order"] * 0.1) for node in meta_genes}

# Initialize Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Metagenome Dashboard"),
    dcc.Graph(id='dag-graph'),
    # Dummy stats section (for demonstration purposes)
    html.Div(id='stats-div', children=[
        html.P("Total Metagenes: 5"),
        html.P("In Basket: 2")
    ]),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])

# Callback to update the graph and stats
@app.callback(Output('dag-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_dag(n):
    edge_x = []
    edge_y = []
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

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if 'Start' in node or 'End' in node:
            node_text.append(f"{node} (base)")
            node_color.append('orange')
        else:
            node_text.append(f"{node} (meta, order {meta_genes[node]['order']})")
            node_color.append('blue')

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

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Metagenome Hierarchy",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig

@app.callback(Output('stats-div', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_stats(n):
    total_metagenes = len(meta_genes)
    in_basket = random.randint(0, 5)  # Example stat, replace with actual logic if needed
    return html.Div([
        html.P(f"Total Metagenes: {total_metagenes}"),
        html.P(f"In Basket: {in_basket}")
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
