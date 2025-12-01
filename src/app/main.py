"""
Main Streamlit Application

This is the main entry point for the Streamlit application with multiple pages.
"""

import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Job Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define pages
PAGES = {
    "Exploration": "pages/1_ exploration.py",
    "Graph Visualization": "pages/2_ graph.py",
    "NLP Analysis": "pages/3_ nlp.py",
    "GNN Models": "pages/4_ gnn.py",
    "Graph-RAG QA": "pages/5_ rag.py",
    "Time Series": "pages/6_ time_series.py",
    "CV Analysis": "pages/7_ cv_analysis.py",
    "Smart CV": "pages/8_ smart_cv.py"  # Added Smart CV page
}

def main():
    """Main application function"""
    st.title("📊 Job Market Intelligence Dashboard")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # App description
    st.markdown("""
    This dashboard provides comprehensive analysis of the data job market using:
    - **NLP** for job classification and skill extraction
    - **Graph Networks** for relationship analysis
    - **GNN Models** for recommendations
    - **Graph-RAG** for question answering
    - **Time Series Analysis** for trend detection
    - **CV Analysis** for personalized recommendations
    - **Smart CV** for OCR-based enhancement
    
    Use the sidebar to navigate between different analysis modules.
    """)
    
    # Display app architecture diagram
    st.markdown("### System Architecture")
    st.image("https://raw.githubusercontent.com/your-org/your-repo/main/assets/architecture.png", 
             caption="Job Analysis System Architecture", use_column_width=True)
    
    # Project information
    st.markdown("### Project Overview")
    st.markdown("""
    This project analyzes job postings to provide insights into:
    - 📈 **Market Trends**: Salary distributions, technology popularity
    - 🔍 **Skill Analysis**: Required skills, skill gaps
    - 🤖 **AI Models**: Classification, recommendation systems
    - 📊 **Data Visualization**: Interactive graphs and charts
    """)
    
    # Instructions
    st.markdown("### Getting Started")
    st.markdown("""
    1. Start with **Exploration** to understand the dataset
    2. View **Graph Visualization** to see job-skill relationships
    3. Explore **NLP Analysis** for job classification and skill extraction
    4. Check **GNN Models** for recommendations and predictions
    5. Use **Graph-RAG QA** to ask questions about the job market
    6. Analyze **Time Series** for trend detection
    7. Upload your **CV** for personalized analysis
    8. Try **Smart CV** for OCR-based enhancement
    """)

if __name__ == "__main__":
    main()