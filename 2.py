import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import hashlib
import numpy as np
from itertools import combinations
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 1. Data loading and preparation
def load_graph_data(file_path):
    """
    Loads and prepares data for a GNN model with features from the dataset and classification.
    """
    data = pd.read_csv(file_path)

    # Extract node features from the dataset columns
    feature_columns = [
        'num_variables', 'num_initialized_vars', 'num_nonzero_initialized_vars', 'num_uint8', 
        'num_uint256', 'num_loops', 'num_payable_functions', 'num_mappings', 'num_arrays', 
        'gas_cost', 'optimization_level'
    ]
    
    # Convert features to numpy array
    node_features = data[feature_columns].values

    # Encode labels (assuming you have a classification column in your dataset)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['optimization_level'])  # Ensure you have a 'classification' column

    # Create fully connected relationships (edges) - You can modify this to your use case
    num_nodes = len(data)
    edges = list(combinations(range(num_nodes), 2))
    random.shuffle(edges)
    subset_edges = edges[:num_nodes * 5]  # Limit to 5 connections per node on average
    edge_index = torch.tensor(subset_edges, dtype=torch.long).T

    # Convert data to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y), len(np.unique(y))

# 2. GAT model definition
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        self.conv2 = GATConv(hidden_dim * 8, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. Training and evaluation
def train_model(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_mask = torch.rand(len(data.y)) < 0.8
    test_mask = ~train_mask

    epoch_losses = []
    train_accuracies = []
    test_accuracies = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())  # Store loss values

        model.eval()
        pred = out.argmax(dim=1)
        train_acc = (pred[train_mask] == data.y[train_mask]).sum().item() / train_mask.sum().item()
        test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Calculate precision, recall, and F1 scores
        y_true = data.y[test_mask].numpy()
        y_pred = pred[test_mask].numpy()
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
    
    return epoch_losses, train_accuracies, test_accuracies, precision_scores, recall_scores, f1_scores, out, test_mask 

# 4. Load data and train GAT model
file_path = "dataset-1.csv"  # Replace with your file
graph_data, num_classes = load_graph_data(file_path)

input_dim = graph_data.x.size(1)
hidden_dim = 64

# GAT
print("\nTraining GAT Model...")
gat_model = GATModel(input_dim, hidden_dim, num_classes)
epoch_losses, train_accuracies, test_accuracies, precision_scores, recall_scores, f1_scores, out, test_mask = train_model(gat_model, graph_data)

# 5. Display results as graphs

# 1. Learning curve (Loss over epochs)
def plot_loss(epoch_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label="Loss", color='blue')
    plt.title("Learning Curve (Loss)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(epoch_losses)

# 2. Training and test accuracy
def plot_accuracy(train_accuracies, test_accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", color='green')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy", color='orange')
    plt.title("Accuracy Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_accuracy(train_accuracies, test_accuracies)

# 3. Precision, Recall, and F1-score
def plot_scores(precision_scores, recall_scores, f1_scores):
    epochs = range(1, len(precision_scores) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, precision_scores, label="Precision", color='blue')
    plt.plot(epochs, recall_scores, label="Recall", color='green')
    plt.plot(epochs, f1_scores, label="F1-score", color='red')
    plt.title("Performance Scores Evolution")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_scores(precision_scores, recall_scores, f1_scores)

# 4. Confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.show()

y_true = graph_data.y[test_mask].numpy()
y_pred = out.argmax(dim=1)[test_mask].numpy()
labels = ["Class 1", "Class 2", "Class 3"]  # Adjust according to your classes
plot_confusion_matrix(y_true, y_pred, labels)

# 5. Node features distribution
plt.figure(figsize=(8, 5))
plt.hist(graph_data.x.numpy().flatten(), bins=50, color='purple', alpha=0.7)
plt.title("Node Features Distribution")
plt.xlabel("Feature Values")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 6. Graph visualization
import networkx as nx
def visualize_graph(edge_index, labels):
    g = nx.Graph()
    edges = edge_index.numpy().T
    g.add_edges_from(edges)
    plt.figure(figsize=(8, 8))
    nx.draw(g, with_labels=True, node_color=labels, cmap=plt.cm.viridis, node_size=100, font_size=8)
    plt.title("Graph Visualization")
    plt.show()

visualize_graph(graph_data.edge_index, graph_data.y.numpy())

# 7. Accuracy by number of neighbors
node_degrees = torch.bincount(graph_data.edge_index[0])  # Number of neighbors per node
accuracies_by_degree = []

# Ensure that test_mask is applied correctly (convert to CPU tensor if needed, for consistency)
test_mask = test_mask.cpu()  # Convert to CPU tensor if needed

# Determine the correct number of nodes (this can be the length of graph_data.x)
num_nodes = graph_data.x.shape[0]

# Ensure test_mask has the same number of nodes
assert test_mask.shape[0] == num_nodes, f"Mismatch in number of nodes: test_mask has {test_mask.shape[0]} but graph has {num_nodes}."

# Convert y_pred and y_true to tensors if they are numpy arrays
if isinstance(y_pred, np.ndarray):
    y_pred = torch.tensor(y_pred)

if isinstance(y_true, np.ndarray):
    y_true = torch.tensor(y_true)

# Now apply .cpu() if needed (ensure both tensors are on CPU)
y_pred = y_pred.cpu()  # Convert to CPU tensor if needed
y_true = y_true.cpu()  # Convert to CPU tensor if needed

# Loop over unique degrees and calculate accuracy by degree
for degree in torch.unique(node_degrees):
    # Mask to filter nodes with the same degree
    degree_mask = (node_degrees == degree)

    # Apply test_mask to get the subset of nodes to evaluate
    filtered_degree_mask = degree_mask[:num_nodes] & test_mask[:num_nodes]

    if filtered_degree_mask.sum() > 0:
        accuracy = (y_pred[filtered_degree_mask] == y_true[filtered_degree_mask]).sum() / filtered_degree_mask.sum()
        accuracies_by_degree.append((degree.item(), accuracy))

# Prepare and plot the accuracies by degree
if accuracies_by_degree:  # Ensure there are results before plotting
    degrees, accuracies = zip(*accuracies_by_degree)
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, accuracies, marker='o', linestyle='-', color='orange')
    plt.title("Accuracy by Number of Neighbors (Degree)")
    plt.xlabel("Degree (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
