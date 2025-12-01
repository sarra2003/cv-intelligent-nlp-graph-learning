"""
Graph Visualization Page for Job Analysis Dashboard

This page provides interactive visualization of the job-company-skill graph.
"""

import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Graph Visualization",
    page_icon="🕸️",
    layout="wide"
)

def load_graph():
    """Load the job graph"""
    try:
        # Try multiple possible paths
        possible_paths = [
            '../../models/job_graph.pkl',
            '../models/job_graph.pkl',
            'models/job_graph.pkl'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    G = pickle.load(f)
                return G
        
        st.error("Graph file not found. Please run the graph construction pipeline first.")
        return None
    except Exception as e:
        st.error(f"Error loading graph: {e}")
        return None

def main():
    """Main function for the graph visualization page"""
    st.title("🕸️ Job-Company-Skill Graph Visualization")
    
    # Load graph
    G = load_graph()
    if G is None:
        return
    
    # Sidebar controls
    st.sidebar.header("Graph Controls")
    
    # Node type filter
    node_types = ['all', 'job', 'company', 'skill']
    selected_node_type = st.sidebar.selectbox(
        "Filter by Node Type",
        options=node_types,
        index=0
    )
    
    # Number of nodes to display
    max_nodes = st.sidebar.slider(
        "Maximum Nodes to Display",
        min_value=10,
        max_value=500,
        value=100
    )
    
    # Filter graph based on selections
    if selected_node_type != 'all':
        # Get nodes of selected type
        filtered_nodes = [n for n, attr in G.nodes(data=True) 
                         if attr.get('node_type') == selected_node_type]
        
        # Limit to max_nodes
        if len(filtered_nodes) > max_nodes:
            filtered_nodes = filtered_nodes[:max_nodes]
        
        # Create subgraph
        G_filtered = G.subgraph(filtered_nodes)
    else:
        # Sample nodes if graph is too large
        if len(G.nodes()) > max_nodes:
            nodes_sample = list(G.nodes())[:max_nodes]
            G_filtered = G.subgraph(nodes_sample)
        else:
            G_filtered = G
    
    # Display graph metrics
    st.subheader("Graph Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Nodes", len(G.nodes()))
    col2.metric("Total Edges", len(G.edges()))
    col3.metric("Filtered Nodes", len(G_filtered.nodes()))
    col4.metric("Filtered Edges", len(G_filtered.edges()))
    
    # Node type distribution
    node_types_count = {}
    for node, attr in G_filtered.nodes(data=True):
        node_type = attr.get('node_type', 'unknown')
        node_types_count[node_type] = node_types_count.get(node_type, 0) + 1
    
    st.bar_chart(node_types_count)
    
    # Display graph information
    st.subheader("Graph Information")
    
    # Node details
    st.markdown("#### Node Types:")
    for node_type, count in node_types_count.items():
        st.write(f"- {node_type.capitalize()}: {count} nodes")
    
    # Sample nodes
    st.markdown("#### Sample Nodes:")
    sample_nodes = list(G_filtered.nodes())[:10]
    node_data = []
    for node in sample_nodes:
        attr = dict(G_filtered.nodes[node])
        node_data.append({
            'Node ID': node,
            'Type': attr.get('node_type', 'unknown'),
            'Attributes': str({k: v for k, v in attr.items() if k != 'node_type'})
        })
    
    st.dataframe(pd.DataFrame(node_data))
    
    # Sample edges
    st.markdown("#### Sample Edges:")
    sample_edges = list(G_filtered.edges())[:10]
    edge_data = []
    for edge in sample_edges:
        source, target = edge
        source_type = G_filtered.nodes[source].get('node_type', 'unknown')
        target_type = G_filtered.nodes[target].get('node_type', 'unknown')
        edge_data.append({
            'Source': f"{source} ({source_type})",
            'Target': f"{target} ({target_type})"
        })
    
    st.dataframe(pd.DataFrame(edge_data))
    
    # Community detection results (if available)
    st.subheader("Community Detection")
    try:
        with open('../../outputs/communities.json', 'r') as f:
            communities = json.load(f)
        
        st.info(f"Detected {len(communities)} communities in the graph")
        
        # Display community information
        for comm_id, nodes in list(communities.items())[:5]:  # Show first 5 communities
            with st.expander(f"Community {comm_id} ({len(nodes)} nodes)"):
                # Sample nodes in community
                sample_nodes = nodes[:10]  # Show first 10 nodes
                node_info = []
                for node in sample_nodes:
                    if node in G_filtered.nodes():
                        attr = G_filtered.nodes[node]
                        node_info.append({
                            'Node': node,
                            'Type': attr.get('node_type', 'unknown'),
                            'Title': attr.get('job_title', '') if attr.get('node_type') == 'job' else ''
                        })
                st.dataframe(pd.DataFrame(node_info))
    except FileNotFoundError:
        st.info("Community detection results not available. Run the graph construction pipeline to generate them.")

if __name__ == "__main__":
    main()