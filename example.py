import pandas as pd
import numpy as np
from minigng import MiniGNG

from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx

# load data.
df = pd.read_csv("/path/to/mnist_train.csv", header=None)
X = df.to_numpy(dtype=np.float32)[:, 1:]
y = df[0].to_numpy()

# init GNG.
gng = MiniGNG(
    max_units=150, 
    n_epochs=25, 
    max_edge_age=50, 
    sample=.2, 
    untangle=True, 
    untangle_net_size=0)

# Traing GNG.
gng.fit(X, y=y)

predictions, unit_ids = gng.predict(X, return_unit_ids=True)
labels = dict(zip(unit_ids, predictions))

# plot the graph using Networkx and Graphviz.
edge_weight = lambda e: np.linalg.norm(e.source.prototype - e.target.prototype)
nodes = [i for i, _ in enumerate(gng.units)]
edges = [(gng.units.index(e.source), gng.units.index(e.target), {'weight': edge_weight(e)}) for e in gng.edges]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

nx.draw(G, pos=nx.nx_pydot.graphviz_layout(G, prog="neato"), node_size=50, labels=labels, node_color='white')