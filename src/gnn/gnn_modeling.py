"""
Graph Neural Network (GNN) Modeling Module

This module implements GNN models for:
1. Node classification (job role prediction)
2. Link prediction (skill recommendation)
3. Community detection
4. Graph embeddings

The module uses PyTorch Geometric for GNN implementations.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import pickle
import json
import os
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Check if torch geometric is available
try:
    import torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not installed. Please install with: pip install torch-geometric")

class GCN(torch.nn.Module):
    """Graph Convolutional Network for node classification"""
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    """Graph Attention Network for node classification"""
    def __init__(self, num_features, hidden_dim, num_classes, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

def load_graph(graph_path: str = 'models/job_graph.pkl') -> nx.Graph:
    """Load graph from pickle file"""
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def create_node_features(G: nx.Graph) -> torch.Tensor:
    """
    Create node features for GNN
    
    Args:
        G (nx.Graph): Input graph
        
    Returns:
        torch.Tensor: Node features tensor
    """
    print("Creating node features...")
    
    # For this example, we'll create simple one-hot encoded features
    # In practice, you would use more sophisticated features like embeddings
    
    nodes = list(G.nodes())
    node_to_id = {node: i for i, node in enumerate(nodes)}
    
    # Simple degree-based features
    degrees = [G.degree(node) for node in nodes]
    max_degree = max(degrees) if degrees else 1
    
    # Normalize degrees
    normalized_degrees = [d / max_degree for d in degrees]
    
    # Create feature matrix (for demo, we'll use degree as feature)
    features = torch.tensor([[degree] for degree in normalized_degrees], dtype=torch.float)
    
    print(f"Created features for {len(nodes)} nodes")
    return features, node_to_id

def create_edge_index(G: nx.Graph, node_to_id: dict) -> torch.Tensor:
    """
    Create edge index tensor for PyG
    
    Args:
        G (nx.Graph): Input graph
        node_to_id (dict): Mapping from node names to IDs
        
    Returns:
        torch.Tensor: Edge index tensor
    """
    print("Creating edge index...")
    
    edges = list(G.edges())
    edge_index = torch.tensor([[node_to_id[src], node_to_id[dst]] for src, dst in edges], dtype=torch.long).t().contiguous()
    
    print(f"Created edge index with {edge_index.shape[1]} edges")
    return edge_index

def prepare_node_classification_data(G: nx.Graph, node_to_id: dict) -> tuple:
    """
    Prepare data for node classification task
    
    Args:
        G (nx.Graph): Input graph
        node_to_id (dict): Mapping from node names to IDs
        
    Returns:
        tuple: (labels, train_mask, test_mask)
    """
    print("Preparing node classification data...")
    
    nodes = list(G.nodes())
    labels = []
    
    # Create labels based on node type for demo
    for node in nodes:
        attr = G.nodes[node]
        node_type = attr.get('node_type', 'unknown')
        
        if node_type == 'job':
            labels.append(0)
        elif node_type == 'company':
            labels.append(1)
        elif node_type == 'skill':
            labels.append(2)
        else:
            labels.append(3)  # unknown
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Create train/test masks
    num_nodes = len(nodes)
    train_size = int(0.8 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_indices = torch.randperm(num_nodes)[:train_size]
    test_indices = torch.randperm(num_nodes)[train_size:]
    
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    
    print(f"Prepared labels for {num_nodes} nodes")
    print(f"Train set size: {train_mask.sum().item()}")
    print(f"Test set size: {test_mask.sum().item()}")
    
    return labels, train_mask, test_mask

def train_gnn_model(model, data, epochs=100, lr=0.01):
    """
    Train GNN model
    
    Args:
        model: GNN model
        data: PyG data object
        epochs (int): Number of training epochs
        lr (float): Learning rate
        
    Returns:
        list: Training losses
    """
    print("Training GNN model...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')
    
    return losses

def evaluate_model(model, data):
    """
    Evaluate model performance
    
    Args:
        model: Trained GNN model
        data: PyG data object
        
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating model...")
    
    model.eval()
    _, pred = model(data).max(dim=1)
    
    # Training accuracy
    train_acc = accuracy_score(
        data.y[data.train_mask].cpu().numpy(),
        pred[data.train_mask].cpu().numpy()
    )
    
    # Test accuracy
    test_acc = accuracy_score(
        data.y[data.test_mask].cpu().numpy(),
        pred[data.test_mask].cpu().numpy()
    )
    
    # F1 scores
    train_f1 = f1_score(
        data.y[data.train_mask].cpu().numpy(),
        pred[data.train_mask].cpu().numpy(),
        average='weighted'
    )
    
    test_f1 = f1_score(
        data.y[data.test_mask].cpu().numpy(),
        pred[data.test_mask].cpu().numpy(),
        average='weighted'
    )
    
    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'train_f1': train_f1,
        'test_f1': test_f1
    }
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training F1: {train_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    
    return metrics

def generate_graph_embeddings(model, data):
    """
    Generate graph embeddings from trained model
    
    Args:
        model: Trained GNN model
        data: PyG data object
        
    Returns:
        torch.Tensor: Node embeddings
    """
    print("Generating graph embeddings...")
    
    model.eval()
    with torch.no_grad():
        embeddings = model.conv1(data.x, data.edge_index)
        embeddings = F.relu(embeddings)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def main():
    """
    Main function to run the GNN modeling pipeline
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("Skipping GNN modeling due to missing dependencies")
        return
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load graph
    G = load_graph()
    
    # Create features
    features, node_to_id = create_node_features(G)
    
    # Create edge index
    edge_index = create_edge_index(G, node_to_id)
    
    # Prepare classification data
    labels, train_mask, test_mask = prepare_node_classification_data(G, node_to_id)
    
    # Create PyG data object
    data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)
    
    # Initialize and train GCN model
    print("\n=== Training GCN Model ===")
    gcn_model = GCN(num_features=features.shape[1], hidden_dim=16, num_classes=4)
    gcn_losses = train_gnn_model(gcn_model, data, epochs=100)
    
    # Evaluate GCN model
    gcn_metrics = evaluate_model(gcn_model, data)
    
    # Initialize and train GAT model
    print("\n=== Training GAT Model ===")
    gat_model = GAT(num_features=features.shape[1], hidden_dim=8, num_classes=4)
    gat_losses = train_gnn_model(gat_model, data, epochs=100)
    
    # Evaluate GAT model
    gat_metrics = evaluate_model(gat_model, data)
    
    # Generate embeddings
    gcn_embeddings = generate_graph_embeddings(gcn_model, data)
    
    # Save models and embeddings
    torch.save(gcn_model.state_dict(), 'models/gcn_model.pth')
    torch.save(gat_model.state_dict(), 'models/gat_model.pth')
    torch.save(gcn_embeddings, 'models/gcn_embeddings.pt')
    
    # Save metrics
    results = {
        'gcn_metrics': gcn_metrics,
        'gat_metrics': gat_metrics
    }
    
    with open('models/gnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nGNN modeling pipeline completed!")
    print("Models and results saved to:")
    print("- models/gcn_model.pth")
    print("- models/gat_model.pth")
    print("- models/gcn_embeddings.pt")
    print("- models/gnn_results.json")

if __name__ == "__main__":
    main()