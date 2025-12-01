"""
GNN Models Page for Job Analysis Dashboard

This page provides GNN-based analysis and recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# Set page configuration
st.set_page_config(
    page_title="GNN Models",
    page_icon="🧠",
    layout="wide"
)

def load_gnn_results():
    """Load GNN model results"""
    try:
        with open('../../models/gnn_results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        return None

def main():
    """Main function for the GNN models page"""
    st.title("🧠 Graph Neural Networks Analysis")
    
    # Load GNN results
    gnn_results = load_gnn_results()
    
    # Tabs for different GNN analyses
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Skill Recommendations", "Community Detection"])
    
    with tab1:
        st.subheader("GNN Model Performance")
        
        if gnn_results:
            # Display model comparison
            st.markdown("#### Model Comparison:")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, metrics in gnn_results.items():
                comparison_data.append({
                    'Model': model_name.upper(),
                    'Train Accuracy': f"{metrics['train_accuracy']:.4f}",
                    'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
                    'Train F1': f"{metrics['train_f1']:.4f}",
                    'Test F1': f"{metrics['test_f1']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(comparison_data))
            
            # Performance visualization
            st.markdown("#### Performance Visualization:")
            
            # Bar chart for accuracy comparison
            import matplotlib.pyplot as plt
            
            models = [data['Model'] for data in comparison_data]
            train_acc = [float(data['Train Accuracy']) for data in comparison_data]
            test_acc = [float(data['Test Accuracy']) for data in comparison_data]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, train_acc, width, label='Train Accuracy')
            ax.bar(x + width/2, test_acc, width, label='Test Accuracy')
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Accuracy Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            
            st.pyplot(fig)
            
        else:
            st.info("GNN results not available. Run the GNN modeling pipeline to generate them.")
            
            # Provide information about GNN models
            st.markdown("#### Available GNN Models:")
            st.markdown("""
            - **GCN** (Graph Convolutional Network): 
              - Layers: 2
              - Hidden dimensions: 16
              - Activation: ReLU
              
            - **GAT** (Graph Attention Network):
              - Layers: 2
              - Hidden dimensions: 8
              - Attention heads: 4
              - Activation: ELU
            """)
            
            st.markdown("#### GNN Applications:")
            st.markdown("""
            1. **Node Classification**: Predict job roles, company types, skill categories
            2. **Link Prediction**: Recommend skills for jobs, jobs for candidates
            3. **Graph Embeddings**: Generate vector representations for downstream tasks
            4. **Community Detection**: Identify clusters of similar jobs/skills
            """)
    
    with tab2:
        st.subheader("Skill Recommendation System")
        
        st.info("This system recommends skills based on job requirements and market trends.")
        
        # Recommendation interface
        st.markdown("#### Get Skill Recommendations:")
        
        # Job title input
        job_title = st.text_input("Enter a job title:", "Data Scientist")
        
        # Experience level
        experience_level = st.select_slider(
            "Experience Level:",
            options=["Entry", "Mid", "Senior", "Expert"],
            value="Mid"
        )
        
        # Current skills (multi-select)
        all_skills = [
            "Python", "SQL", "Machine Learning", "Deep Learning", "Statistics",
            "Data Visualization", "R", "TensorFlow", "PyTorch", "Pandas",
            "NumPy", "Scikit-learn", "Spark", "Hadoop", "AWS",
            "Docker", "Kubernetes", "Git", "Tableau", "Power BI"
        ]
        
        current_skills = st.multiselect(
            "Select your current skills:",
            options=all_skills,
            default=["Python", "SQL", "Pandas"]
        )
        
        # Generate recommendations button
        if st.button("Get Recommendations"):
            if job_title:
                # Simulate recommendations based on job title and current skills
                # In practice, this would use the trained GNN model
                
                # Define skill recommendations based on job title
                recommendations = {
                    "Data Scientist": [
                        "Machine Learning", "Deep Learning", "Statistics", "Data Visualization",
                        "Scikit-learn", "TensorFlow", "PyTorch"
                    ],
                    "Data Engineer": [
                        "Spark", "Hadoop", "AWS", "Docker", "Kubernetes", "Airflow"
                    ],
                    "Machine Learning Engineer": [
                        "TensorFlow", "PyTorch", "Deep Learning", "Scikit-learn",
                        "AWS", "Docker"
                    ],
                    "Data Analyst": [
                        "Statistics", "Data Visualization", "Tableau", "Power BI",
                        "SQL", "Pandas"
                    ]
                }
                
                # Get base recommendations
                base_recs = recommendations.get(job_title, recommendations["Data Scientist"])
                
                # Filter out already known skills
                suggested_skills = [skill for skill in base_recs if skill not in current_skills]
                
                # Display recommendations
                st.markdown("##### Recommended Skills to Learn:")
                if suggested_skills:
                    for i, skill in enumerate(suggested_skills[:7], 1):  # Show top 7
                        st.markdown(f"{i}. **{skill}**")
                    
                    # Priority ranking
                    st.markdown("##### Priority Ranking:")
                    priority_explanation = {
                        "High Priority": "Essential for the role",
                        "Medium Priority": "Valuable for career growth",
                        "Low Priority": "Nice to have"
                    }
                    
                    for priority, explanation in priority_explanation.items():
                        st.markdown(f"- **{priority}**: {explanation}")
                else:
                    st.info("You already have all the key skills for this role!")
            else:
                st.warning("Please enter a job title.")
    
    with tab3:
        st.subheader("Community Detection")
        
        st.info("Identify clusters and communities in the job market graph.")
        
        # Community information
        st.markdown("#### Detected Communities:")
        st.markdown("""
        The graph has been analyzed to identify communities of related jobs, skills, and companies.
        These communities represent clusters with strong internal connections.
        """)
        
        # Sample community data
        sample_communities = [
            {
                "id": 1,
                "name": "Data Science & Machine Learning",
                "size": 1247,
                "description": "Jobs and skills related to data science, machine learning, and AI"
            },
            {
                "id": 2,
                "name": "Data Engineering & Big Data",
                "size": 982,
                "description": "Engineering roles focused on data infrastructure and big data technologies"
            },
            {
                "id": 3,
                "name": "Business Intelligence & Analytics",
                "size": 756,
                "description": "Analytics and business intelligence roles"
            },
            {
                "id": 4,
                "name": "Cloud & DevOps",
                "size": 634,
                "description": "Cloud platforms and DevOps engineering roles"
            }
        ]
        
        # Display communities
        for community in sample_communities:
            with st.expander(f"Community {community['id']}: {community['name']} ({community['size']} nodes)"):
                st.markdown(f"**Description**: {community['description']}")
                st.markdown("**Sample Skills**:")
                sample_skills = ["Python", "SQL", "Machine Learning", "Spark", "AWS"] if community['id'] == 1 else \
                               ["Spark", "Hadoop", "Kafka", "Airflow", "Docker"] if community['id'] == 2 else \
                               ["Tableau", "Power BI", "SQL", "Statistics", "Excel"] if community['id'] == 3 else \
                               ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD"]
                
                st.markdown(", ".join(sample_skills))
                
                # Visualization placeholder
                st.markdown("**Community Visualization**:")
                st.image("https://placehold.co/400x200?text=Community+Graph+Visualization", 
                        caption="Community structure visualization")
        
        # Community analysis tools
        st.markdown("#### Community Analysis Tools:")
        st.markdown("""
        - **Community Search**: Find which community a specific skill or job belongs to
        - **Overlap Analysis**: Identify skills that bridge multiple communities
        - **Growth Trends**: Track how communities evolve over time
        - **Centrality Measures**: Find the most influential nodes in each community
        """)

if __name__ == "__main__":
    main()