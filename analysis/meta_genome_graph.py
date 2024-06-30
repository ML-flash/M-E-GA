import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Dataset (updated with provided data)
dataset = {3068971186: 'Start', 205742900: 'End', 2632741828: 'R', 1842982710: 'L', 475484659: 'U', 115861103: 'D', 548432130: 'F', 1761407013: 'B', 493800275: (2632741828, 548432130), 1150380693: (2632741828, 493800275), 4176702928: (2632741828, 548432130, 1842982710), 3560278447: (2632741828, 2632741828, 548432130), 2696007376: (475484659, 548432130), 145565013: (548432130, 493800275), 1624044850: (2632741828, 548432130, 475484659), 2954834188: (548432130, 1842982710), 2045035353: (2632741828, 493800275, 1842982710), 4154328180: (548432130, 548432130), 3828246499: (493800275, 2632741828, 548432130), 1413683241: (493800275, 2632741828), 4061434644: (548432130, 2632741828), 4101144470: (475484659, 475484659), 2475751335: (475484659, 548432130, 2632741828), 3733400472: (493800275, 2632741828, 548432130, 1842982710), 2308678168: (493800275, 548432130, 2045035353), 1395126785: (548432130, 475484659), 2176268583: (493800275, 2632741828, 548432130, 548432130), 1312413709: (2176268583, 493800275, 2632741828), 3439133634: (548432130, 2632741828, 493800275), 1812219875: (2308678168, 493800275, 2632741828, 548432130), 3143051122: (1413683241, 1413683241, 2632741828, 548432130), 3806142345: (493800275, 548432130, 2045035353, 548432130), 3731042954: (548432130, 493800275, 548432130, 2632741828), 3539326365: (548432130, 493800275, 2632741828, 548432130), 274176478: (493800275, 548432130, 2632741828), 872131287: (2632741828, 548432130, 2632741828, 548432130), 2081842553: (548432130, 493800275, 548432130, 2045035353), 1318069296: (475484659, 2632741828, 548432130), 1559412798: (548432130, 3828246499), 1489338083: (493800275, 4101144470, 2632741828), 581508734: (548432130, 493800275, 2632741828), 425741723: (493800275, 475484659), 1500835147: (548432130, 475484659, 493800275), 3437725142: (2632741828, 425741723, 548432130), 2204322135: (475484659, 2632741828, 548432130, 548432130), 4068419733: (1761407013, 425741723)}



# Create a directed graph
G = nx.DiGraph()

# Process the dataset to build the graph
for gene, contents in dataset.items():
    if isinstance(contents, str):
        # Base gene
        G.add_node(gene, label=contents, order=0)
    elif isinstance(contents, tuple):
        max_order = 0
        for parent in contents:
            if parent in G.nodes:
                max_order = max(max_order, G.nodes[parent]['order'] + 1)
            G.add_edge(parent, gene)
        G.add_node(gene, order=max_order)

# Assign labels to nodes for visualization
labels = {node: data['label'] if 'label' in data else node for node, data in G.nodes(data=True)}

# Assign positions for layered visualization
pos = {}
layer_nodes = defaultdict(list)
for node, data in G.nodes(data=True):
    order = data['order']
    layer_nodes[order].append(node)

# Set y-coordinates based on order
y = 0
for order in sorted(layer_nodes):
    x = 0
    for node in layer_nodes[order]:
        pos[node] = (x, y)
        x += 1
    y += 1

# Draw the graph
plt.figure(figsize=(20, 15))
nx.draw(G, pos, with_labels=True, labels=labels, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
plt.title("Gene Relationship Graph with Orders")
plt.show()
