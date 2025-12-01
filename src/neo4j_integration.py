"""
Neo4j Integration Module for Job Market Graph
"""
import os
import json
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jGraphManager:
    """Manages Neo4j graph database operations for job market data"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """
        Initialize Neo4j connection
        
        Args:
            uri (str): Neo4j database URI
            user (str): Username
            password (str): Password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        
    def connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
            
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
            
    def create_indexes(self):
        """Create indexes for better query performance"""
        if not self.driver:
            logger.error("No Neo4j connection")
            return
            
        with self.driver.session() as session:
            # Create indexes
            session.run("CREATE INDEX IF NOT EXISTS FOR (j:Job) ON (j.title)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (s:Skill) ON (s.name)")
            logger.info("Created Neo4j indexes")
            
    def load_graph_from_file(self, graph_file: str = "models/job_graph.pkl") -> nx.Graph:
        """
        Load NetworkX graph from file
        
        Args:
            graph_file (str): Path to graph pickle file
            
        Returns:
            nx.Graph: Loaded graph
        """
        if os.path.exists(graph_file):
            graph = nx.read_gpickle(graph_file)
            logger.info(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph
        else:
            logger.error(f"Graph file not found: {graph_file}")
            return None
            
    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        if not self.driver:
            logger.error("No Neo4j connection")
            return
            
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Cleared Neo4j database")
            
    def import_graph_to_neo4j(self, graph: nx.Graph):
        """
        Import NetworkX graph to Neo4j database
        
        Args:
            graph (nx.Graph): NetworkX graph to import
        """
        if not self.driver or not graph:
            logger.error("No Neo4j connection or graph")
            return
            
        with self.driver.session() as session:
            # Clear existing data
            self.clear_database()
            
            # Create nodes
            logger.info("Creating nodes...")
            for node, data in graph.nodes(data=True):
                node_type = data.get('type', 'Node')
                properties = {k: v for k, v in data.items() if k != 'type'}
                properties['id'] = str(node)  # Ensure ID is string
                
                if node_type == 'job':
                    session.run(
                        "CREATE (j:Job $props)",
                        props=properties
                    )
                elif node_type == 'company':
                    session.run(
                        "CREATE (c:Company $props)",
                        props=properties
                    )
                elif node_type == 'skill':
                    session.run(
                        "CREATE (s:Skill $props)",
                        props=properties
                    )
                else:
                    session.run(
                        "CREATE (n:Node $props)",
                        props=properties
                    )
                    
            # Create relationships
            logger.info("Creating relationships...")
            for source, target, data in graph.edges(data=True):
                relationship_type = data.get('type', 'RELATED_TO').upper()
                properties = {k: v for k, v in data.items() if k != 'type'}
                
                session.run(
                    f"""
                    MATCH (a {{id: $source}}), (b {{id: $target}})
                    CREATE (a)-[r:{relationship_type} $props]->(b)
                    """,
                    source=str(source),
                    target=str(target),
                    props=properties
                )
                
            logger.info(f"Imported graph to Neo4j: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
    def query_graph(self, cypher_query: str, parameters: Dict = None) -> List[Dict]:
        """
        Execute a Cypher query on the graph database
        
        Args:
            cypher_query (str): Cypher query to execute
            parameters (Dict): Query parameters
            
        Returns:
            List[Dict]: Query results
        """
        if not self.driver:
            logger.error("No Neo4j connection")
            return []
            
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
            
    def get_job_recommendations(self, skills: List[str], limit: int = 10) -> List[Dict]:
        """
        Get job recommendations based on skills
        
        Args:
            skills (List[str]): List of skills
            limit (int): Maximum number of recommendations
            
        Returns:
            List[Dict]: Job recommendations
        """
        query = """
        MATCH (s:Skill)-[:REQUIRES]->(j:Job)-[:OFFERED_BY]->(c:Company)
        WHERE s.name IN $skills
        RETURN j.title AS job_title, c.name AS company, 
               count(s) AS matching_skills
        ORDER BY matching_skills DESC
        LIMIT $limit
        """
        
        return self.query_graph(query, {"skills": skills, "limit": limit})
        
    def get_skill_recommendations(self, current_skills: List[str], limit: int = 10) -> List[Dict]:
        """
        Get skill recommendations based on current skills
        
        Args:
            current_skills (List[str]): List of current skills
            limit (int): Maximum number of recommendations
            
        Returns:
            List[Dict]: Skill recommendations
        """
        query = """
        MATCH (s1:Skill)-[:REQUIRES]->(j:Job)<-[:REQUIRES]-(s2:Skill)
        WHERE s1.name IN $current_skills AND NOT s2.name IN $current_skills
        RETURN s2.name AS recommended_skill, count(*) AS frequency
        ORDER BY frequency DESC
        LIMIT $limit
        """
        
        return self.query_graph(query, {"current_skills": current_skills, "limit": limit})
        
    def get_market_trends(self) -> Dict[str, Any]:
        """
        Get market trends from the graph
        
        Returns:
            Dict[str, Any]: Market trends data
        """
        trends = {}
        
        # Top skills
        query = """
        MATCH (s:Skill)<-[:REQUIRES]-(j:Job)
        RETURN s.name AS skill, count(j) AS frequency
        ORDER BY frequency DESC
        LIMIT 20
        """
        trends['top_skills'] = self.query_graph(query)
        
        # Top companies
        query = """
        MATCH (c:Company)<-[:OFFERED_BY]-(j:Job)
        RETURN c.name AS company, count(j) AS job_count
        ORDER BY job_count DESC
        LIMIT 20
        """
        trends['top_companies'] = self.query_graph(query)
        
        # Job distribution by type
        query = """
        MATCH (j:Job)
        RETURN j.type AS job_type, count(j) AS count
        ORDER BY count DESC
        """
        trends['job_types'] = self.query_graph(query)
        
        return trends

def main():
    """Main function to demonstrate Neo4j integration"""
    # Initialize Neo4j manager
    neo4j_manager = Neo4jGraphManager()
    
    try:
        # Connect to Neo4j
        neo4j_manager.connect()
        
        # Create indexes
        neo4j_manager.create_indexes()
        
        # Load graph from file
        graph = neo4j_manager.load_graph_from_file()
        
        if graph:
            # Import graph to Neo4j
            neo4j_manager.import_graph_to_neo4j(graph)
            
            # Example queries
            logger.info("Running example queries...")
            
            # Get job recommendations
            recommendations = neo4j_manager.get_job_recommendations(['Python', 'Machine Learning'])
            logger.info(f"Job recommendations: {recommendations}")
            
            # Get skill recommendations
            skill_recs = neo4j_manager.get_skill_recommendations(['Python', 'SQL'])
            logger.info(f"Skill recommendations: {skill_recs}")
            
            # Get market trends
            trends = neo4j_manager.get_market_trends()
            logger.info(f"Market trends: {trends}")
            
    except Exception as e:
        logger.error(f"Error in Neo4j integration: {e}")
    finally:
        # Close connection
        neo4j_manager.close()

if __name__ == "__main__":
    main()