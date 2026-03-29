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
from sklearn.metrics import confusion_matrix

# 1. Chargement et préparation des données
def hash_to_features(hash_values):
    """
    Convertit une liste de hash en caractéristiques numériques.
    """
    features = []
    for hash_val in hash_values:
        hash_int = int(hashlib.md5(hash_val.encode()).hexdigest(), 16)  # MD5 converti en entier
        features.append([hash_int % 10000])  # Réduction à une plage de valeurs
    return np.array(features)

def load_graph_data(file_path):
    """
    Charge et prépare les données pour un modèle GNN avec deux colonnes : hash et classification.
    """
    data = pd.read_csv(file_path)

    # Extraire les caractéristiques des nœuds à partir de la colonne hash
    node_features = hash_to_features(data['hash'])

    # Encoder les étiquettes
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['classification'])

    # Créer les relations (edges) de manière entièrement connectée
    num_nodes = len(data)
    edges = list(combinations(range(num_nodes), 2))
    random.shuffle(edges)
    subset_edges = edges[:num_nodes * 5]  # Limiter à 5 connexions par nœud en moyenne
    edge_index = torch.tensor(subset_edges, dtype=torch.long).T

    # Convertir les données en tenseurs
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y), len(np.unique(y))

# 2. Définition du modèle GAT
class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8, concat=True)
        self.conv2 = GATConv(hidden_dim * 8, output_dim, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 3. Entraînement et évaluation
def train_model(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_mask = torch.rand(len(data.y)) < 0.8
    test_mask = ~train_mask

    epoch_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())  # Stocker les valeurs de loss

        model.eval()
        pred = out.argmax(dim=1)
        train_acc = (pred[train_mask] == data.y[train_mask]).sum().item() / train_mask.sum().item()
        test_acc = (pred[test_mask] == data.y[test_mask]).sum().item() / test_mask.sum().item()

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")
    
    return epoch_losses, train_accuracies, test_accuracies, out, test_mask 

# 4. Chargement des données et entraînement du modèle GAT
file_path = "reduced_dataset.csv"  # Remplacez par votre fichier
graph_data, num_classes = load_graph_data(file_path)

input_dim = graph_data.x.size(1)
hidden_dim = 64

# GAT
print("\nTraining GAT Model...")
gat_model = GATModel(input_dim, hidden_dim, num_classes)
epoch_losses, train_accuracies, test_accuracies, out, test_mask = train_model(gat_model, graph_data)

# 5. Affichage des résultats sous forme de graphiques

# 1. Courbe d'apprentissage (Loss au fil des époques)
def plot_loss(epoch_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label="Loss", color='blue')
    plt.title("Courbe d'apprentissage (Loss)")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(epoch_losses)

# 2. Précision sur l’ensemble d'entraînement et de test
def plot_accuracy(train_accuracies, test_accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Précision Entraînement", color='green')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Précision Test", color='orange')
    plt.title("Évolution de la précision")
    plt.xlabel("Époques")
    plt.ylabel("Précision")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_accuracy(train_accuracies, test_accuracies)

# 3. Matrice de confusion
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.show()

y_true = graph_data.y[test_mask].numpy()
y_pred = out.argmax(dim=1)[test_mask].numpy()
labels = ["Classe 1", "Classe 2", "Classe 3"]  # Adaptez selon vos classes
plot_confusion_matrix(y_true, y_pred, labels)

# 4. Distribution des caractéristiques des nœuds
plt.figure(figsize=(8, 5))
plt.hist(graph_data.x.numpy().flatten(), bins=50, color='purple', alpha=0.7)
plt.title("Distribution des caractéristiques des nœuds")
plt.xlabel("Valeurs des caractéristiques")
plt.ylabel("Fréquence")
plt.grid(True)
plt.show()

# 5. Visualisation du graphe
import networkx as nx
def visualize_graph(edge_index, labels):
    g = nx.Graph()
    edges = edge_index.numpy().T
    g.add_edges_from(edges)
    plt.figure(figsize=(8, 8))
    nx.draw(g, with_labels=True, node_color=labels, cmap=plt.cm.viridis, node_size=100, font_size=8)
    plt.title("Visualisation du graphe")
    plt.show()

visualize_graph(graph_data.edge_index, graph_data.y.numpy())

# 6. Précision en fonction du nombre de voisins
node_degrees = torch.bincount(graph_data.edge_index[0])  # Nombre de voisins par nœud
accuracies_by_degree = []
for degree in torch.unique(node_degrees):
    mask = (node_degrees[test_mask] == degree).numpy()
    if mask.sum() > 0:
        accuracy = (y_pred[mask] == y_true[mask]).sum() / mask.sum()
        accuracies_by_degree.append((degree.item(), accuracy))

degrees, accuracies = zip(*accuracies_by_degree)
plt.figure(figsize=(8, 5))
plt.plot(degrees, accuracies, marker='o', color='red')
plt.title("accuracy")
plt.xlabel("Nombre de voisins")
plt.ylabel("Précision")
plt.grid(True)
plt.show()

# 7. Comparaison des embeddings des nœuds avant et après entraînement
def plot_embeddings(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", s=10)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()

# Avant entraînement :
plot_embeddings(graph_data.x.numpy(), graph_data.y.numpy(), "Embeddings des nœuds (avant entraînement)")

# Après entraînement :
gat_model.eval()
with torch.no_grad():
    embeddings = gat_model.conv1(graph_data.x, graph_data.edge_index).numpy()
plot_embeddings(embeddings, graph_data.y.numpy(), "Embeddings des nœuds (après entraînement)")
