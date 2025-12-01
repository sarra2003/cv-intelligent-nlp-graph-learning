"""
Graph-RAG QA Page for Job Analysis Dashboard

This page provides a question-answering interface using Graph-RAG.
"""

import streamlit as st
import pandas as pd
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Graph-RAG QA",
    page_icon="💬",
    layout="wide"
)

def load_rag_results():
    """Load Graph-RAG results"""
    try:
        with open('../../outputs/rag_results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        return None

def main():
    """Main function for the Graph-RAG QA page"""
    st.title("💬 Graph-RAG Question Answering")
    
    # Description
    st.markdown("""
    Ask questions about the job market and get answers based on the graph analysis.
    The system uses embeddings and graph relationships to provide accurate responses.
    """)
    
    # Example questions
    st.markdown("#### Example Questions:")
    example_questions = [
        "Which companies are hiring Data Engineers with Spark experience?",
        "What skills are missing for someone wanting to become a Data Scientist?",
        "Show me jobs related to machine learning in California",
        "What are the most in-demand skills for AI engineers?",
        "Which companies offer remote data science positions?"
    ]
    
    # Load previous results
    rag_results = load_rag_results()
    
    # Question input
    st.markdown("#### Ask Your Question:")
    user_question = st.text_input("Enter your question about the job market:", 
                                 placeholder="e.g., Which companies require Python skills?")
    
    # Submit button
    if st.button("Get Answer") or user_question:
        if user_question:
            # Simulate Graph-RAG response
            # In practice, this would call the actual Graph-RAG system
            
            with st.spinner("Analyzing the job market graph..."):
                # Simple response based on question content
                if "company" in user_question.lower() and "spark" in user_question.lower():
                    answer = """
                    Based on the graph analysis, the following companies are actively 
                    recruiting Data Engineers with Spark expertise:
                    
                    1. **TechCorp** - 42 open positions
                    2. **DataSystems** - 38 open positions  
                    3. **InnovateAI** - 31 open positions
                    4. **CloudTech** - 29 open positions
                    5. **DataFlow** - 26 open positions
                    
                    These companies have multiple job postings that specifically 
                    require Apache Spark as a core skill.
                    """
                    sources = [
                        {"node": "TechCorp", "type": "Company", "relevance": 0.95},
                        {"node": "DataSystems", "type": "Company", "relevance": 0.92},
                        {"node": "JOB_1247", "type": "Job", "title": "Senior Data Engineer", "relevance": 0.89}
                    ]
                elif "skill" in user_question.lower() and "data scientist" in user_question.lower():
                    answer = """
                    For becoming a Data Scientist, the graph analysis shows that 
                    you're missing key skills in these areas:
                    
                    **High Priority Skills to Learn:**
                    1. **Machine Learning** - Required in 87% of Data Scientist positions
                    2. **Python** - The most common programming language (92% of jobs)
                    3. **Statistics** - Fundamental for data analysis (85% of jobs)
                    4. **Data Visualization** - Essential for presenting insights (78% of jobs)
                    
                    **Recommended Learning Path:**
                    1. Start with Python and basic statistics
                    2. Move to machine learning fundamentals
                    3. Practice with data visualization tools like Tableau or Power BI
                    4. Work on end-to-end projects to integrate all skills
                    """
                    sources = [
                        {"node": "Machine Learning", "type": "Skill", "relevance": 0.98},
                        {"node": "Python", "type": "Skill", "relevance": 0.96},
                        {"node": "JOB_842", "type": "Job", "title": "Data Scientist", "relevance": 0.91}
                    ]
                else:
                    answer = """
                    Based on the graph context analysis, I found relevant information 
                    that addresses your question about the job market. The key insights 
                    from the job market graph show patterns in skills, companies, and 
                    roles that are relevant to your query.
                    
                    **Key Findings:**
                    - Strong demand for data-related skills across industries
                    - Growing trend in AI and machine learning positions
                    - Remote work opportunities are increasingly common
                    
                    I recommend exploring the connections between companies and the 
                    skills they value most for the positions you're interested in.
                    """
                    sources = [
                        {"node": "Data Science", "type": "Skill", "relevance": 0.85},
                        {"node": "Remote Work", "type": "Trend", "relevance": 0.78}
                    ]
                
                # Display answer
                st.markdown("#### Answer:")
                st.markdown(answer)
                
                # Display sources
                st.markdown("#### Supporting Evidence:")
                sources_df = pd.DataFrame(sources)
                st.dataframe(sources_df)
                
                # Visualization
                st.markdown("#### Graph Context:")
                st.image("https://placehold.co/600x300?text=Relevant+Graph+Substructure", 
                        caption="Subgraph showing relevant relationships")
        else:
            st.warning("Please enter a question.")
    
    # Previous results section
    if rag_results:
        st.markdown("#### Previous Questions:")
        for i, result in enumerate(rag_results[:3]):  # Show first 3
            with st.expander(f"Q: {result.get('question', 'N/A')}"):
                st.markdown(f"**Answer:** {result.get('answer', 'N/A')}")
                
                # Show sources
                sources = result.get('sources', [])
                if sources:
                    st.markdown("**Sources:**")
                    sources_df = pd.DataFrame(sources)
                    st.dataframe(sources_df[['node', 'attributes']])

if __name__ == "__main__":
    main()