import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from minigng import MiniGNG

dataset = "/path/to/dataset.csv"
output = "graph.png"

# load data.
df = pd.read_csv(dataset, header=None)

# assuming labels are in the first column.
X = df.to_numpy(dtype=np.float32)[:, 1:]
y = df[0].to_numpy()

# init GNG.
gng = MiniGNG(
    max_units=100, 
    n_epochs=10,
    max_edge_age=50, 
    sample=.2, 
    untangle=True, 
    max_size_connect=3)

# Traing GNG.
gng.fit(X, y=y)

predictions, unit_ids = gng.predict(X, return_unit_ids=True)
labels = dict(zip(unit_ids, predictions))

# plot the graph using Networkx and Graphviz.
nodes = [i for i, _ in enumerate(gng.units)]
edges = [(gng.units.index(e.source), gng.units.index(e.target), {}) for e in gng.edges]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

layout = nx.nx_pydot.graphviz_layout(G, prog="neato")
nx.draw(
    G,
    pos=layout,
    node_size=50,
    labels=labels,
    node_color="white",
    edge_color="gray",
    width=.5)
plt.savefig(output)