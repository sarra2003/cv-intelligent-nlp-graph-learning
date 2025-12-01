"""
Interactive Graph Visualization with PyVis

This module creates an interactive visualization of the job-company-skill graph
using PyVis library.
"""

import pandas as pd
import networkx as nx
from pyvis.network import Network
import json
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_graph(graph_path: str = 'models/job_graph.pkl') -> nx.Graph:
    """
    Load graph from pickle file
    
    Args:
        graph_path (str): Path to the graph pickle file
        
    Returns:
        nx.Graph: Loaded graph
    """
    print(f"Loading graph from {graph_path}...")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"Loaded graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G

def create_interactive_network(G: nx.Graph, max_nodes: int = 200) -> Network:
    """
    Create interactive network visualization
    
    Args:
        G (nx.Graph): Input graph
        max_nodes (int): Maximum number of nodes to include
        
    Returns:
        Network: PyVis network object
    """
    print("Creating interactive network visualization...")
    
    # Create PyVis network
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=False
    )
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    # Sample nodes if graph is too large
    if len(G.nodes()) > max_nodes:
        print(f"Sampling {max_nodes} nodes for visualization...")
        nodes_sample = list(G.nodes())[:max_nodes]
        G_sub = G.subgraph(nodes_sample)
    else:
        G_sub = G
    
    # Add nodes with colors based on type
    for node, attr in G_sub.nodes(data=True):
        node_type = attr.get('node_type', 'unknown')
        
        if node_type == 'job':
            color = '#ADD8E6'  # Light blue
            size = 15
            title = f"Job: {attr.get('job_title', node)}"
        elif node_type == 'company':
            color = '#90EE90'  # Light green
            size = 25
            title = f"Company: {node}"
        elif node_type == 'skill':
            color = '#FFB6C1'  # Light pink
            size = 10
            title = f"Skill: {node}"
        else:
            color = '#D3D3D3'  # Light gray
            size = 10
            title = f"Unknown: {node}"
        
        net.add_node(
            node,
            label=node[:20] + "..." if len(node) > 20 else node,
            title=title,
            color=color,
            size=size
        )
    
    # Add edges
    for source, target in G_sub.edges():
        net.add_edge(source, target)
    
    return net

def save_interactive_visualization(net: Network, filename: str = 'outputs/interactive_graph.html') -> None:
    """
    Save interactive visualization to HTML file
    
    Args:
        net (Network): PyVis network object
        filename (str): Output filename
    """
    print(f"Saving interactive visualization to {filename}...")
    net.save_graph(filename)
    print("Interactive visualization saved successfully!")

def analyze_graph_metrics(G: nx.Graph) -> None:
    """
    Analyze and display graph metrics
    
    Args:
        G (nx.Graph): Input graph
    """
    print("\nGraph Metrics:")
    print(f"Number of nodes: {len(G.nodes())}")
    print(f"Number of edges: {len(G.edges())}")
    
    # Count node types
    node_types = {}
    for node, attr in G.nodes(data=True):
        node_type = attr.get('node_type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNode type distribution:")
    for node_type, count in node_types.items():
        print(f"  {node_type}: {count}")
    
    # Calculate degree centrality for top nodes
    if len(G.nodes()) < 10000:  # Only for smaller graphs
        degree_centrality = nx.degree_centrality(G)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most connected nodes:")
        for node, centrality in top_nodes:
            attr = G.nodes[node]
            node_type = attr.get('node_type', 'unknown')
            if node_type == 'job':
                label = f"Job: {attr.get('job_title', node)[:30]}"
            elif node_type == 'company':
                label = f"Company: {node}"
            elif node_type == 'skill':
                label = f"Skill: {node}"
            else:
                label = node
            print(f"  {label[:40]}: {centrality:.4f}")

def main():
    """
    Main function to create interactive graph visualization
    """
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load graph
    G = load_graph()
    
    # Analyze graph metrics
    analyze_graph_metrics(G)
    
    # Create interactive network
    net = create_interactive_network(G)
    
    # Save visualization
    save_interactive_visualization(net)
    
    print("\nInteractive visualization pipeline completed!")
    print("Open outputs/interactive_graph.html in your browser to view the interactive graph.")

if __name__ == "__main__":
    main()