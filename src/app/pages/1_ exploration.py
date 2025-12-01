"""
Exploration Page for Job Analysis Dashboard

This page provides exploratory data analysis of job postings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Job Exploration",
    page_icon="🔍",
    layout="wide"
)

def load_data():
    """Load job data"""
    try:
        # Try multiple possible paths
        possible_paths = [
            '../../data/data_jobs_clean.csv',
            '../data/data_jobs_clean.csv',
            'data/data_jobs_clean.csv',
            '../../data/data_jobs.csv',
            '../data/data_jobs.csv',
            'data/data_jobs.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
        
        st.error("Data file not found. Please run the data processing pipeline first.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def main():
    """Main function for the exploration page"""
    st.title("🔍 Job Market Exploration")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Job title filter
    job_titles = df['job_title_clean'].unique()
    selected_titles = st.sidebar.multiselect(
        "Select Job Titles",
        options=job_titles,
        default=job_titles[:5] if len(job_titles) > 5 else job_titles
    )
    
    # Company filter
    companies = df['company'].unique()
    selected_companies = st.sidebar.multiselect(
        "Select Companies",
        options=companies,
        default=companies[:5] if len(companies) > 5 else companies
    )
    
    # Filter data
    filtered_df = df.copy()
    if selected_titles:
        filtered_df = filtered_df[filtered_df['job_title_clean'].isin(selected_titles)]
    if selected_companies:
        filtered_df = filtered_df[filtered_df['company'].isin(selected_companies)]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", len(df))
    col2.metric("Filtered Jobs", len(filtered_df))
    col3.metric("Companies", df['company'].nunique())
    col4.metric("Unique Job Titles", df['job_title_clean'].nunique())
    
    # Dataset overview
    st.subheader("Dataset Overview")
    st.dataframe(filtered_df.head(10))
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Job Titles", "Companies", "Skills", "Locations"])
    
    with tab1:
        st.subheader("Job Title Analysis")
        
        # Top job titles
        top_titles = filtered_df['job_title_clean'].value_counts().head(20)
        
        # Matplotlib bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top_titles)), top_titles.values)
        ax.set_yticks(range(len(top_titles)))
        ax.set_yticklabels(top_titles.index)
        ax.set_xlabel("Number of Postings")
        ax.set_title("Top 20 Job Titles")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
        
        # Detailed table
        st.dataframe(top_titles.reset_index().rename(columns={'index': 'Job Title', 'job_title_clean': 'Count'}))
    
    with tab2:
        st.subheader("Company Analysis")
        
        # Top companies
        top_companies = filtered_df['company'].value_counts().head(20)
        
        # Interactive Plotly chart
        fig = px.bar(
            x=top_companies.values,
            y=top_companies.index,
            orientation='h',
            title="Top 20 Companies by Job Postings",
            labels={'x': 'Number of Postings', 'y': 'Company'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Skills Analysis")
        
        # Extract skills
        all_skills = []
        for skills_str in filtered_df['skills_list'].dropna():
            try:
                skills_list = json.loads(skills_str) if isinstance(skills_str, str) else skills_str
                if isinstance(skills_list, list):
                    all_skills.extend([skill.lower().strip() for skill in skills_list])
            except:
                continue
        
        # Top skills
        skill_counts = Counter(all_skills)
        top_skills = dict(skill_counts.most_common(20))
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top_skills)), list(top_skills.values()))
        ax.set_yticks(range(len(top_skills)))
        ax.set_yticklabels(list(top_skills.keys()))
        ax.set_xlabel("Frequency")
        ax.set_title("Top 20 Skills")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
        
        # Skills table
        skills_df = pd.DataFrame(list(top_skills.items()), columns=['Skill', 'Count'])
        st.dataframe(skills_df)
    
    with tab4:
        st.subheader("Location Analysis")
        
        if 'location' in filtered_df.columns:
            # Top locations
            top_locations = filtered_df['location'].value_counts().head(20)
            
            # Map visualization (if location data includes coordinates)
            fig = px.bar(
                x=top_locations.values,
                y=top_locations.index,
                orientation='h',
                title="Top 20 Locations",
                labels={'x': 'Number of Postings', 'y': 'Location'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Location data not available in the dataset.")
    
    # Data quality section
    st.subheader("Data Quality")
    
    # Missing values
    missing_values = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_values / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percent
    })
    
    st.dataframe(missing_df[missing_df['Missing Values'] > 0])
    
    # Data types
    st.subheader("Data Types")
    st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

if __name__ == "__main__":
    main()