import torch
from torch_geometric.data import Data
from model import VTXGNN
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

matplotlib.use("TkAgg")

# Example usage
num_node_features = 3  # Number of node features
num_classes = 2  # Number of classes

model = VTXGNN(3, 3, 0.2)

# Example data
# Note: You should replace these with your actual graph data
nodes = torch.tensor([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=torch.float)
edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous()
edge_weights = torch.tensor([0.2, 0.3], dtype=torch.float)

cons = Data(x=nodes, edge_index=edges, edge_attr=edge_weights)

edges = torch.tensor([[0, 2], [1, 2]], dtype=torch.long).t().contiguous()
edge_weights = torch.tensor([0.2, 0.3], dtype=torch.float)

sply = Data(x=cons.x, edge_index=edges, edge_attr=edge_weights)

# Forward pass
output = model(cons.x, cons.edge_index, cons.edge_attr, sply.edge_index, sply.edge_attr)
print(output)
# Assuming 'data' is your PyG Data object
G = to_networkx(output, to_undirected=True)

# Draw the graph
nx.draw(G, with_labels=True, node_color="lightblue", font_weight="bold")
plt.savefig("./graph.png")
