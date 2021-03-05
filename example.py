import pandas as pd
import numpy as np
from minigng import MiniGNG

from scipy import stats
import matplotlib.pyplot as plt
import networkx as nx

# load data.
df = pd.read_csv("/path/to/mnist_train.csv", header=None)
training = df.to_numpy(dtype=np.float32)[:, 1:]

# traing GNG.
gng = MiniGNG(
    max_units=150, 
    n_epochs=25, 
    max_edge_age=50, 
    sample=.2, 
    untangle=True, 
    untangle_net_size=0)

gng.fit(training)

# get unit ids for each data sample.
predictions = np.array(gng.transform(training))

# assign a label to each unit given a majority vote (scipy mode).
y = df[0].to_numpy()
unit_ids = range(len(gng.units))
nodes = {i: stats.mode(y[predictions == i]) for i in unit_ids}
labels = {k: v.mode[0] for k, v in nodes.items()}

# plot the graph using Networkx and Graphviz.
nodes = [i for i, _ in enumerate(gng.units)]
edges = [(gng.units.index(e.source), gng.units.index(e.target)) for e in gng.edges]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

nx.draw(G, pos=nx.nx_pydot.graphviz_layout(G, prog="neato"), node_size=50, labels=labels, node_color='white')