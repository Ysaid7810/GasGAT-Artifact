import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import hashlib
import numpy as np
from itertools import combinations
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Function to hash to features
def hash_to_features(hash_values):
    features = []
    for hash_val in hash_values:
        hash_int = int(hashlib.md5(hash_val.encode()).hexdigest(), 16)  # MD5 converted to integer
        features.append([hash_int % 10000])  # Reduces to a range of values
    return np.array(features)

# Function to load data
def load_graph_data(file_path):
    data = pd.read_csv(file_path)
    node_features = hash_to_features(data['hash'])
    y = pd.factorize(data['classification'])[0]
    num_nodes = len(data)
    edges = list(combinations(range(num_nodes), 2))
    random.shuffle(edges)
    subset_edges = edges[:num_nodes * 5]
    edge_index = torch.tensor(subset_edges, dtype=torch.long).T
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)

# GAT Model
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        self.conv2 = GATConv(hidden_dim * 8, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Plot embeddings
def plot_embeddings(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

# Load data
file_path = "reduced_dataset.csv"  # Replace with your file
graph_data = load_graph_data(file_path)
input_dim = graph_data.x.size(1)
hidden_dim = 64
num_classes = len(torch.unique(graph_data.y))

# Initialize and process embeddings
gat_model = GATModel(input_dim, hidden_dim, num_classes)
before_training_embeddings = gat_model(graph_data.x, graph_data.edge_index).detach().numpy()

# Train model
optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.01)
for epoch in range(100):
    gat_model.train()
    optimizer.zero_grad()
    out = gat_model(graph_data.x, graph_data.edge_index)
    loss = F.cross_entropy(out, graph_data.y)
    loss.backward()
    optimizer.step()

# Extract embeddings after training
after_training_embeddings = gat_model(graph_data.x, graph_data.edge_index).detach().numpy()

# Plot embeddings before and after training
plot_embeddings(before_training_embeddings, graph_data.y.numpy(), "Embeddings Before Training")
plot_embeddings(after_training_embeddings, graph_data.y.numpy(), "Embeddings After Training")
