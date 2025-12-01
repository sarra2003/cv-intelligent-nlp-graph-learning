"""
CV Analysis Page for Job Analysis Dashboard

This page provides CV profiling and improvement suggestions.
"""

import streamlit as st
import pandas as pd
import json
import os

# Set page configuration
st.set_page_config(
    page_title="CV Analysis",
    page_icon="📄",
    layout="wide"
)

def load_cv_results():
    """Load CV analysis results"""
    try:
        with open('../../outputs/cv_analysis_results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        return None

def main():
    """Main function for the CV analysis page"""
    st.title("📄 CV Analysis & Improvement")
    
    # Description
    st.markdown("""
    Upload your CV to get a detailed analysis and personalized improvement suggestions.
    The system compares your skills with market requirements and identifies gaps.
    """)
    
    # File upload
    st.markdown("#### Upload Your CV:")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Simulate CV analysis
        # In practice, this would process the actual uploaded file
        
        with st.spinner("Analyzing your CV..."):
            # Display success message
            st.success("CV successfully uploaded and analyzed!")
            
            # Load sample results
            cv_results = load_cv_results()
            
            if cv_results:
                # Display match score
                comp_results = cv_results.get('comparison_results', {})
                match_score = comp_results.get('overall_match_score', 0)
                
                st.markdown("#### Match Score:")
                st.progress(match_score / 100)
                st.markdown(f"**{match_score:.1f}%** match with target job requirements")
                
                # Skills analysis
                st.markdown("#### Skills Analysis:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Matching Skills:")
                    matching_required = comp_results.get('matching_required_skills', [])
                    matching_preferred = comp_results.get('matching_preferred_skills', [])
                    
                    st.markdown("**Required Skills:**")
                    for skill in matching_required:
                        st.markdown(f"- ✅ {skill}")
                    
                    st.markdown("**Preferred Skills:**")
                    for skill in matching_preferred:
                        st.markdown(f"- ✅ {skill}")
                
                with col2:
                    st.markdown("##### Missing Skills:")
                    missing_required = comp_results.get('missing_required_skills', [])
                    missing_preferred = comp_results.get('missing_preferred_skills', [])
                    
                    st.markdown("**Required Skills:**")
                    for skill in missing_required:
                        st.markdown(f"- ❌ {skill}")
                    
                    st.markdown("**Preferred Skills:**")
                    for skill in missing_preferred:
                        st.markdown(f"- ⚠️ {skill}")
                
                # Improvement suggestions
                st.markdown("#### Improvement Suggestions:")
                suggestions = cv_results.get('improvement_suggestions', [])
                
                for i, suggestion in enumerate(suggestions, 1):
                    st.markdown(f"{i}. {suggestion}")
                
                # Improved CV template
                st.markdown("#### Improved CV Template:")
                improved_cv = cv_results.get('improved_cv_template', '')
                
                with st.expander("View Improved CV Template"):
                    st.text_area("Improved CV", improved_cv, height=400)
                
                # Download button
                st.download_button(
                    label="Download Improved CV Template",
                    data=improved_cv,
                    file_name="improved_cv_template.txt",
                    mime="text/plain"
                )
            else:
                # Sample analysis
                st.markdown("#### Sample Analysis:")
                
                # Sample match score
                st.markdown("#### Match Score:")
                st.progress(65)
                st.markdown("**65.0%** match with target job requirements")
                
                # Sample skills analysis
                st.markdown("#### Skills Analysis:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Matching Skills:")
                    st.markdown("**Required Skills:**")
                    st.markdown("- ✅ Python")
                    st.markdown("- ✅ SQL")
                    st.markdown("- ✅ Machine Learning")
                    st.markdown("- ✅ Statistics")
                    
                    st.markdown("**Preferred Skills:**")
                    st.markdown("- ✅ Pandas")
                    st.markdown("- ✅ Scikit-learn")
                    st.markdown("- ⚠️ TensorFlow")
                    st.markdown("- ⚠️ AWS")
                
                with col2:
                    st.markdown("##### Missing Skills:")
                    st.markdown("**Required Skills:**")
                    st.markdown("- ❌ Deep Learning")
                    st.markdown("- ❌ Data Visualization")
                    
                    st.markdown("**Preferred Skills:**")
                    st.markdown("- ⚠️ PyTorch")
                    st.markdown("- ⚠️ Docker")
                    st.markdown("- ⚠️ Kubernetes")
                    st.markdown("- ⚠️ Spark")
                
                # Sample suggestions
                st.markdown("#### Improvement Suggestions:")
                sample_suggestions = [
                    "Add projects demonstrating deep learning skills",
                    "Include data visualization examples in your portfolio",
                    "Gain experience with PyTorch through online courses",
                    "Consider AWS certification to boost your profile",
                    "Quantify your achievements with specific metrics"
                ]
                
                for i, suggestion in enumerate(sample_suggestions, 1):
                    st.markdown(f"{i}. {suggestion}")
    else:
        # Information about CV analysis
        st.info("Upload your CV to get started with the analysis.")
        
        # Features
        st.markdown("#### Features:")
        st.markdown("""
        - **Skill Extraction**: Automatically identify skills from your CV
        - **Gap Analysis**: Compare your skills with market requirements
        - **Match Scoring**: Get a quantitative measure of your fit
        - **Personalized Suggestions**: Receive tailored improvement recommendations
        - **CV Template**: Get an improved CV template based on analysis
        """)
        
        # How it works
        st.markdown("#### How It Works:")
        st.markdown("""
        1. **Upload**: Submit your CV in PDF format
        2. **Process**: System extracts skills and experiences
        3. **Analyze**: Compare with job market requirements
        4. **Recommend**: Get personalized improvement suggestions
        5. **Improve**: Use the enhanced CV template
        """)

if __name__ == "__main__":
    main()