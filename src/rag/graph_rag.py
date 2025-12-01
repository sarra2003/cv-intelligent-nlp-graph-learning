"""
Graph-RAG Implementation for Job Market Intelligence
"""

import os
import json
import numpy as np
import networkx as nx
import pickle
from typing import List, Tuple, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer

# Try to import Together AI
try:
    from together import Together
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False
    print("Together.ai package not available. LLM functionality will be limited.")

class GraphRAG:
    """
    Graph-RAG system for job market intelligence
    """
    
    def __init__(self, graph_path: str = "models/job_graph.pkl", 
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Graph-RAG system
        
        Args:
            graph_path (str): Path to the job knowledge graph
            model_name (str): Sentence transformer model name
        """
        # Load graph
        self.graph = self.load_graph(graph_path) if os.path.exists(graph_path) else None
        
        # Initialize sentence transformer
        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(model_name)
        
        # Initialize Together AI client if available
        self.client = None
        self.llm_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Updated model name
        if TOGETHER_AVAILABLE:
            api_key = os.getenv("TOGETHER_API_KEY")
            if api_key:
                self.client = Together(api_key=api_key)  # type: ignore
            else:
                print("TOGETHER_API_KEY not found in environment variables")
        
        # Initialize FAISS index and embeddings
        self.index = None
        self.node_embeddings = None
    
    def load_graph(self, graph_path: str) -> nx.Graph:
        """
        Load knowledge graph from file
        
        Args:
            graph_path (str): Path to the graph file
            
        Returns:
            nx.Graph: Loaded knowledge graph
        """
        print(f"Loading graph from {graph_path}...")
        # Use read_gpickle from networkx
        try:
            # Try to load with networkx read_gpickle
            try:
                # Use getattr to avoid static analysis issues
                read_gpickle_func = getattr(nx, 'read_gpickle')
                graph = read_gpickle_func(graph_path)
            except AttributeError:
                # Alternative method for newer networkx versions
                with open(graph_path, 'rb') as f:
                    graph = pickle.load(f)
            print(f"Loaded graph with {graph.number_of_nodes()} nodes")
            return graph
        except Exception as e:
            print(f"Error loading graph: {e}")
            return nx.Graph()
    
    def prepare_node_texts(self) -> List[str]:
        """
        Prepare text representations of all nodes for embedding
        
        Returns:
            List[str]: Text representation of each node
        """
        print("Preparing node text representations...")
        node_texts = []
        
        if self.graph is None:
            print("No graph loaded")
            return node_texts
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            
            # Create a comprehensive text representation
            text_parts = [f"Node ID: {node_id}"]
            
            # Add all node attributes
            for key, value in node_data.items():
                if value and str(value).strip():
                    text_parts.append(f"{key}: {value}")
            
            # Add neighbor information
            neighbors = list(self.graph.neighbors(node_id))
            if neighbors:
                neighbor_info = ", ".join([str(n) for n in neighbors[:5]])  # Limit to first 5
                text_parts.append(f"Connected to: {neighbor_info}")
            
            node_text = " | ".join(text_parts)
            node_texts.append(node_text)
        
        print(f"Prepared text for {len(node_texts)} nodes")
        return node_texts
    
    def generate_node_embeddings(self, node_texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for all node texts
        
        Args:
            node_texts (List[str]): Text representations of nodes
            
        Returns:
            np.ndarray: Node embeddings
        """
        print("Generating node embeddings...")
        if not node_texts:
            print("No node texts to embed")
            return np.array([])
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(node_texts), batch_size):
            batch = node_texts[i:i+batch_size]
            embeddings = self.model.encode(batch, show_progress_bar=True)
            all_embeddings.append(embeddings)
        
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            print(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings.astype('float32')
        else:
            print("Failed to generate embeddings")
            return np.array([])
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for fast similarity search
        
        Args:
            embeddings (np.ndarray): Node embeddings
        """
        print("Building FAISS index...")
        if embeddings.size == 0:
            print("No embeddings to index")
            return
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        # Ensure embeddings are in the right format
        embeddings_float32 = np.ascontiguousarray(embeddings).astype('float32')
        self.index.add(embeddings_float32)  # type: ignore
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def encode_question(self, question: str) -> np.ndarray:
        """
        Encode a question into embedding
        
        Args:
            question (str): Input question
            
        Returns:
            np.ndarray: Question embedding
        """
        if not question:
            return np.array([])
        
        embedding = self.model.encode([question])
        return embedding.astype('float32')
    
    def search_similar_nodes(self, question_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for nodes similar to the question
        
        Args:
            question_embedding (np.ndarray): Encoded question
            k (int): Number of similar nodes to retrieve
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, distance) tuples
        """
        print("Searching for similar nodes...")
        if self.index is None or question_embedding.size == 0:
            return []
        
        # Search in FAISS index
        # Ensure question embedding is in the right format
        question_embedding_float32 = np.ascontiguousarray(question_embedding).astype('float32')
        distances, indices = self.index.search(question_embedding_float32, k)  # type: ignore
        
        # Map indices back to node IDs
        node_ids = list(self.graph.nodes()) if self.graph is not None else []
        similar_nodes = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(node_ids):
                node_id = node_ids[idx]
                similar_nodes.append((node_id, float(distance)))
        
        return similar_nodes
    
    def extract_graph_context(self, node_ids: List[str]) -> nx.Graph:
        """
        Extract subgraph context around relevant nodes
        
        Args:
            node_ids (List[str]): List of relevant node IDs
            
        Returns:
            nx.Graph: Extracted subgraph
        """
        print("Extracting graph context...")
        if self.graph is None:
            return nx.Graph()
        
        # Create subgraph with relevant nodes and their neighbors
        all_nodes = set(node_ids)
        
        # Add neighbors of relevant nodes
        for node_id in node_ids:
            if self.graph.has_node(node_id):
                neighbors = list(self.graph.neighbors(node_id))
                all_nodes.update(neighbors)
        
        # Extract subgraph
        subgraph = self.graph.subgraph(all_nodes).copy()
        print(f"Extracted subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
        return subgraph
    
    def format_context_for_llm(self, subgraph: nx.Graph) -> str:
        """
        Format subgraph context for LLM consumption
        
        Args:
            subgraph (nx.Graph): Extracted subgraph
            
        Returns:
            str: Formatted context string
        """
        if subgraph.number_of_nodes() == 0:
            return "No relevant context found."
        
        context_lines = []
        
        # Add nodes
        context_lines.append("### Relevant Job Nodes:")
        for node_id in subgraph.nodes():
            node_data = subgraph.nodes[node_id]
            node_info = f"- {node_id}: "
            
            # Add key attributes
            attrs = []
            for key in ['title', 'company', 'location', 'skills', 'description']:
                if key in node_data and node_data[key]:
                    attrs.append(f"{key}='{node_data[key]}'")
            
            node_info += ", ".join(attrs) if attrs else "No attributes"
            context_lines.append(node_info)
        
        # Add relationships
        context_lines.append("\n### Key Relationships:")
        for edge in list(subgraph.edges())[:20]:  # Limit to first 20 edges
            source, target = edge
            edge_data = subgraph.get_edge_data(source, target, default={})
            relation_type = edge_data.get('relation', 'connected_to')
            context_lines.append(f"- {source} --[{relation_type}]--> {target}")
        
        return "\n".join(context_lines)
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using Together LLM
        """
        # Debug: Print context length and first part
        print(f"Context length: {len(context)}")
        print(f"Context preview: {context[:500]}...")
        
        if TOGETHER_AVAILABLE and self.client is not None:
            print("Generating answer using LLM via Together.ai...")
            
            prompt = f"""
You are an AI assistant using a knowledge graph for job market intelligence.

### User Question:
{question}

### Graph Context (nodes and relationships found):
{context}

### Instructions:
- Answer ONLY using the information in the graph context.
- Do NOT hallucinate.
- If the graph context does not contain the answer, say: 
  "I could not find enough information in the graph to answer this question."
- For questions about skills or competencies, provide a clean list format without lengthy explanations.
- Provide structured, clear reasoning.

### Final Answer:
"""

            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a specialized Graph-RAG assistant. For questions about skills or competencies, provide a clean bullet list format without lengthy explanations. Format: '- Skill 1\n- Skill 2\n- Skill 3'"},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=350,
                    temperature=0.2,
                )

                # Extract the answer from the response
                # Handle potential None response
                if response is not None:
                    # Check if response has choices attribute
                    if hasattr(response, 'choices'):
                        choices = getattr(response, 'choices', None)
                        # Check if choices is not None and has elements
                        if choices is not None and len(choices) > 0:
                            choice = choices[0]
                            # Check if choice has message attribute
                            if hasattr(choice, 'message'):
                                message = getattr(choice, 'message', None)
                                # Check if message has content attribute
                                if hasattr(message, 'content'):
                                    content = getattr(message, 'content', None)
                                    # Check if content is not None
                                    if content is not None:
                                        answer = str(content)
                                        print(f"Generated answer: {answer}")
                                        return answer
                
                return "Unable to extract answer from LLM response"

            except Exception as e:
                print(f"Error generating answer with LLM: {str(e)}")
                return f"Error generating answer with LLM: {str(e)}"
        else:
            print("Together.ai not available")
            return "LLM not available. Please install together package and configure API key."
    
    def answer_question(self, question: str, k: int = 10) -> Dict:
        """
        Answer a question using Graph-RAG approach
        
        Args:
            question (str): Input question
            k (int): Number of relevant nodes to retrieve
            
        Returns:
            Dict: Answer and supporting information
        """
        print(f"Answering question: {question}")
        
        # Encode question
        question_embedding = self.encode_question(question)
        if question_embedding.size == 0:
            return {"error": "Failed to encode question"}
        
        # Search for similar nodes
        similar_nodes = self.search_similar_nodes(question_embedding, k)
        if not similar_nodes:
            return {"error": "No similar nodes found"}
        
        # Extract node IDs
        node_ids = [node_id for node_id, _ in similar_nodes]
        
        # Extract graph context
        subgraph = self.extract_graph_context(node_ids)
        
        # Format context
        context = self.format_context_for_llm(subgraph)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Prepare result
        result = {
            "question": question,
            "answer": answer,
            "context": context,
            "similar_nodes": similar_nodes,
            "sources": [
                {
                    "node": node_id,
                    "distance": distance,
                    "attributes": dict(self.graph.nodes[node_id]) if self.graph is not None and self.graph.has_node(node_id) else {}
                }
                for node_id, distance in similar_nodes
            ]
        }
        
        return result
    
    def build_system(self):
        """Build the complete Graph-RAG system"""
        print("Building Graph-RAG system...")
        
        # Prepare node texts
        node_texts = self.prepare_node_texts()
        
        # Generate embeddings
        self.node_embeddings = self.generate_node_embeddings(node_texts)
        if self.node_embeddings.size == 0:
            print("Failed to generate node embeddings")
            return
        
        # Build FAISS index
        self.build_faiss_index(self.node_embeddings)
        
        print("Graph-RAG system built successfully!")

def main():
    """
    Main function to demonstrate Graph-RAG functionality
    """
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize Graph-RAG system
    rag = GraphRAG()
    
    # Build system
    rag.build_system()
    
    # Example questions
    questions = [
        "Which companies are hiring Data Engineers with Spark experience?",
        "What skills are missing for someone wanting to become a Data Scientist?",
        "Show me jobs related to machine learning in California"
    ]
    
    # Answer questions
    results = []
    for question in questions:
        result = rag.answer_question(question)
        results.append(result)
        
        print(f"\nQuestion: {question}")
        print(f"Answer: {result.get('answer', 'No answer generated')}")
        print(f"Sources: {len(result.get('sources', []))} relevant nodes found")
    
    # Save results
    with open('outputs/rag_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nGraph-RAG pipeline completed!")
    print("Results saved to outputs/rag_results.json")

if __name__ == "__main__":
    main()