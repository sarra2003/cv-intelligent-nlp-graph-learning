"""
Graph Construction Module

This module constructs a bipartite graph with three types of nodes:
1. JOB nodes
2. COMPANY nodes
3. SKILL nodes

The module creates edges between:
- Company → Job
- Job → Skill

It also provides visualization and export functionality.
"""

import pandas as pd
import networkx as nx
import json
import pickle
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import warnings
from .neo4j_integration import Neo4jGraphManager
warnings.filterwarnings('ignore')

def create_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Create a bipartite graph from job data
    
    Args:
        df (pd.DataFrame): Dataframe with job data including entities
        
    Returns:
        nx.Graph: Bipartite graph
    """
    print("Creating bipartite graph...")
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with attributes
    print("Adding nodes...")
    
    # Add job nodes
    for idx, row in df.iterrows():
        job_id = f"JOB_{idx}"
        G.add_node(
            job_id,
            node_type="job",
            job_title=row.get('job_title_clean', ''),
            company=row.get('company', ''),
            location=row.get('location', '') if 'location' in row else ''
        )
    
    # Add company nodes
    companies = df['company'].unique()
    for company in companies:
        if pd.notna(company):
            G.add_node(company, node_type="company")
    
    # Add skill nodes
    all_skills = set()
    for skills_str in df['skills_list'].dropna():
        try:
            skills_list = json.loads(skills_str) if isinstance(skills_str, str) else skills_str
            if isinstance(skills_list, list):
                all_skills.update([skill.lower().strip() for skill in skills_list])
        except:
            continue
    
    for skill in all_skills:
        G.add_node(skill, node_type="skill")
    
    print(f"Added {len(G.nodes())} nodes:")
    print(f"  - Job nodes: {len([n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'job'])}")
    print(f"  - Company nodes: {len([n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'company'])}")
    print(f"  - Skill nodes: {len([n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'skill'])}")
    
    # Add edges
    print("Adding edges...")
    
    # Company -> Job edges
    company_edges = 0
    for idx, row in df.iterrows():
        job_id = f"JOB_{idx}"
        company = row.get('company')
        if pd.notna(company) and company in G.nodes():
            G.add_edge(company, job_id)
            company_edges += 1
    
    # Job -> Skill edges
    skill_edges = 0
    for idx, row in df.iterrows():
        job_id = f"JOB_{idx}"
        skills_str = row.get('skills_list')
        if pd.notna(skills_str):
            try:
                skills_list = json.loads(skills_str) if isinstance(skills_str, str) else skills_str
                if isinstance(skills_list, list):
                    for skill in skills_list:
                        skill = skill.lower().strip()
                        if skill in G.nodes():
                            G.add_edge(job_id, skill)
                            skill_edges += 1
            except:
                continue
    
    print(f"Added {company_edges} company-job edges")
    print(f"Added {skill_edges} job-skill edges")
    print(f"Total edges: {len(G.edges())}")
    
    return G

def visualize_graph(G: nx.Graph, max_nodes: int = 50) -> None:
    """
    Visualize the graph using matplotlib
    
    Args:
        G (nx.Graph): Graph to visualize
        max_nodes (int): Maximum number of nodes to visualize
    """
    print("Visualizing graph...")
    
    # Limit nodes for visualization
    if len(G.nodes()) > max_nodes:
        print(f"Graph too large for visualization. Sampling {max_nodes} nodes...")
        nodes_sample = list(G.nodes())[:max_nodes]
        G_sub = G.subgraph(nodes_sample)
    else:
        G_sub = G
    
    # Create layout
    pos = nx.spring_layout(G_sub, k=1, iterations=50)
    
    # Color nodes by type
    node_colors = []
    for node, attr in G_sub.nodes(data=True):
        node_type = attr.get('node_type', 'unknown')
        if node_type == 'job':
            node_colors.append('lightblue')
        elif node_type == 'company':
            node_colors.append('lightgreen')
        elif node_type == 'skill':
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgray')
    
    # Draw graph
    plt.figure(figsize=(16, 12))
    nx.draw(G_sub, pos, node_color=node_colors, with_labels=False, node_size=300, alpha=0.7)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Jobs'),
        Patch(facecolor='lightgreen', label='Companies'),
        Patch(facecolor='lightcoral', label='Skills')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Job-Company-Skill Bipartite Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/graph_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Graph visualization saved to outputs/graph_visualization.png")

def export_to_graphml(G: nx.Graph, filename: str = 'outputs/job_graph.graphml') -> None:
    """
    Export graph to GraphML format
    
    Args:
        G (nx.Graph): Graph to export
        filename (str): Output filename
    """
    print(f"Exporting graph to {filename}...")
    nx.write_graphml(G, filename)
    print("Graph exported successfully!")

def generate_cypher_queries(G: nx.Graph) -> List[str]:
    """
    Generate Cypher queries for Neo4j import
    
    Args:
        G (nx.Graph): Graph to convert
        
    Returns:
        List[str]: List of Cypher queries
    """
    print("Generating Cypher queries for Neo4j...")
    
    queries = []
    
    # Create constraints
    queries.append("CREATE CONSTRAINT ON (j:Job) ASSERT j.id IS UNIQUE;")
    queries.append("CREATE CONSTRAINT ON (c:Company) ASSERT c.name IS UNIQUE;")
    queries.append("CREATE CONSTRAINT ON (s:Skill) ASSERT s.name IS UNIQUE;")
    queries.append("")
    
    # Create nodes
    job_nodes = [(n, attr) for n, attr in G.nodes(data=True) if attr.get('node_type') == 'job']
    company_nodes = [(n, attr) for n, attr in G.nodes(data=True) if attr.get('node_type') == 'company']
    skill_nodes = [(n, attr) for n, attr in G.nodes(data=True) if attr.get('node_type') == 'skill']
    
    # Add job nodes
    for node, attr in job_nodes[:1000]:  # Limit for demo
        job_title = attr.get('job_title', '').replace("'", "\\'")
        company = attr.get('company', '').replace("'", "\\'")
        queries.append(f"CREATE (:Job {{id: '{node}', title: '{job_title}', company: '{company}'}});")
    
    # Add company nodes
    for node, attr in company_nodes:
        # Escape single quotes for Cypher
        escaped_node = node.replace("'", "\\'")
        queries.append(f"CREATE (:Company {{name: '{escaped_node}'}});")
    
    # Add skill nodes
    for node, attr in skill_nodes:
        # Escape single quotes for Cypher
        escaped_node = node.replace("'", "\\'")
        queries.append(f"CREATE (:Skill {{name: '{escaped_node}'}});")
    
    queries.append("")
    
    # Add edges
    edges = list(G.edges())[:2000]  # Limit for demo
    for source, target in edges:
        source_attr = G.nodes[source]
        target_attr = G.nodes[target]
        
        source_type = source_attr.get('node_type')
        target_type = target_attr.get('node_type')
        
        if source_type == 'company' and target_type == 'job':
            # Escape single quotes for Cypher
            escaped_source = source.replace("'", "\\'")
            queries.append(f"MATCH (c:Company {{name: '{escaped_source}'}}), (j:Job {{id: '{target}'}}) CREATE (c)-[:OFFERS]->(j);")
        elif source_type == 'job' and target_type == 'skill':
            # Escape single quotes for Cypher
            escaped_target = target.replace("'", "\\'")
            queries.append(f"MATCH (j:Job {{id: '{source}'}}), (s:Skill {{name: '{escaped_target}'}}) CREATE (j)-[:REQUIRES]->(s);")
        elif source_type == 'job' and target_type == 'company':
            # Escape single quotes for Cypher
            escaped_target = target.replace("'", "\\'")
            queries.append(f"MATCH (j:Job {{id: '{source}'}}), (c:Company {{name: '{escaped_target}'}}) CREATE (c)-[:OFFERS]->(j);")
        elif source_type == 'skill' and target_type == 'job':
            # Escape single quotes for Cypher
            escaped_source = source.replace("'", "\\'")
            queries.append(f"MATCH (s:Skill {{name: '{escaped_source}'}}), (j:Job {{id: '{target}'}}) CREATE (j)-[:REQUIRES]->(s);")
    
    return queries

def save_cypher_queries(queries: List[str], filename: str = 'outputs/graph_import.cypher') -> None:
    """
    Save Cypher queries to file
    
    Args:
        queries (List[str]): List of Cypher queries
        filename (str): Output filename
    """
    print(f"Saving Cypher queries to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(query + '\n')
    print("Cypher queries saved successfully!")

def detect_communities(G: nx.Graph) -> Dict[str, List[str]]:
    """
    Detect communities in the graph using Louvain algorithm
    
    Args:
        G (nx.Graph): Input graph
        
    Returns:
        Dict[str, List[str]]: Communities
    """
    try:
        import community as community_louvain
        
        print("Detecting communities...")
        
        # Create subgraph with only job-skill connections for community detection
        job_skill_edges = [(u, v) for u, v in G.edges() 
                          if (G.nodes[u].get('node_type') == 'job' and G.nodes[v].get('node_type') == 'skill') or
                             (G.nodes[u].get('node_type') == 'skill' and G.nodes[v].get('node_type') == 'job')]
        
        G_sub = nx.Graph()
        G_sub.add_edges_from(job_skill_edges)
        
        # Apply Louvain community detection
        partition = community_louvain.best_partition(G_sub)
        
        # Group nodes by community
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(node)
        
        print(f"Detected {len(communities)} communities")
        return communities
        
    except ImportError:
        print("Community detection requires python-louvain package. Install with: pip install python-louvain")
        return {}

def main():
    """
    Main function to run the graph construction pipeline
    """
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/jobs_with_entities.csv')
    print(f"Loaded {len(df)} job postings with entities")
    
    # Create bipartite graph
    G = create_bipartite_graph(df)
    
    # Save graph
    with open('models/job_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("Graph saved to models/job_graph.pkl")
    
    # Visualize graph
    visualize_graph(G)
    
    # Export to GraphML
    export_to_graphml(G)
    
    # Generate Cypher queries
    cypher_queries = generate_cypher_queries(G)
    save_cypher_queries(cypher_queries)
    
    # Detect communities
    communities = detect_communities(G)
    
    # Save communities
    if communities:
        with open('outputs/communities.json', 'w') as f:
            json.dump(communities, f, indent=2)
        print("Communities saved to outputs/communities.json")
    
    # Integrate with Neo4j
    try:
        neo4j_manager = Neo4jGraphManager()
        neo4j_manager.connect()
        if neo4j_manager.driver:
            neo4j_manager.create_indexes()
            neo4j_manager.import_graph_to_neo4j(G)
            print("Graph successfully imported to Neo4j database")
            neo4j_manager.close()
    except Exception as e:
        print(f"Neo4j integration failed: {e}")
    
    print("\nGraph construction pipeline completed!")
    print("Outputs saved to:")
    print("- models/job_graph.pkl")
    print("- outputs/graph_visualization.png")
    print("- outputs/job_graph.graphml")
    print("- outputs/graph_import.cypher")
    print("- outputs/communities.json")

if __name__ == "__main__":
    main()