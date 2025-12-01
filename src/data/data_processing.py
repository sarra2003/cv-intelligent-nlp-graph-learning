"""
Data Processing Module for Job Analysis Project

This module handles:
1. Loading the raw dataset
2. Cleaning and preprocessing text data
3. Standardizing skills and competencies
4. Parsing skill lists
5. Removing noise and anomalies
6. Exporting cleaned data
"""

import pandas as pd
import numpy as np
import re
import json
from typing import List, Set
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw dataset from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    return df

def clean_text(text: str) -> str:
    """
    Clean and normalize text data
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep letters, numbers, and common punctuation
    text = re.sub(r'[^\w\s\-.,!?;:]', '', text)
    
    # Strip leading/trailing whitespaces
    text = text.strip()
    
    return text

def standardize_skills(skills_str: str) -> List[str]:
    """
    Parse and standardize skills list
    
    Args:
        skills_str (str): String containing skills (comma separated)
        
    Returns:
        List[str]: List of standardized skills
    """
    if pd.isna(skills_str) or skills_str == "":
        return []
    
    # Split by comma
    skills = [skill.strip() for skill in str(skills_str).split(',')]
    
    # Standardize skill names (lowercase, remove extra spaces)
    standardized_skills = []
    for skill in skills:
        # Clean the skill name
        skill = clean_text(skill)
        skill = skill.lower()
        
        # Skip empty skills
        if skill and len(skill) > 1:
            standardized_skills.append(skill)
    
    return standardized_skills

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers and anomalies from the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("Removing outliers...")
    
    # Remove rows with missing job titles
    initial_rows = len(df)
    df = df.dropna(subset=['job_title'])
    print(f"Removed {initial_rows - len(df)} rows with missing job titles")
    
    # Remove duplicate job postings (using actual column names)
    initial_rows = len(df)
    # Use the actual column names from the dataset
    duplicate_columns = ['job_title']
    if 'company_name' in df.columns:
        duplicate_columns.append('company_name')
    if 'job_location' in df.columns:
        duplicate_columns.append('job_location')
    
    df = df.drop_duplicates(subset=duplicate_columns)
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Remove rows with unrealistic salary values (optional)
    # This would depend on the specific salary columns in your dataset
    
    return df

def process_job_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process job descriptions and requirements
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with processed descriptions
    """
    print("Processing job descriptions...")
    
    # Clean job titles
    df['job_title_clean'] = df['job_title'].apply(clean_text)
    
    # Clean job descriptions if they exist
    # Using actual column names from the dataset
    if 'job_description' in df.columns:
        df['job_description_clean'] = df['job_description'].apply(clean_text)
    
    # Process skills column (using actual column name)
    if 'job_skills' in df.columns:
        df['skills_list'] = df['job_skills'].apply(standardize_skills)
    
    # Rename company column to match expected name
    if 'company_name' in df.columns:
        df['company'] = df['company_name']
    
    # Rename location column to match expected name
    if 'job_location' in df.columns:
        df['location'] = df['job_location']
    
    return df

def export_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Export cleaned data to CSV
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        output_path (str): Output file path
    """
    print(f"Exporting cleaned data to {output_path}...")
    
    # Create a copy to avoid modifying the original
    df_export = df.copy()
    
    # Convert list columns to JSON strings for CSV export
    if 'skills_list' in df_export.columns:
        df_export['skills_list'] = df_export['skills_list'].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    
    df_export.to_csv(output_path, index=False)
    print("Data exported successfully!")

def main():
    """
    Main function to run the data processing pipeline
    """
    # Define file paths
    input_file = "data/data_jobs.csv"
    output_file = "data/data_jobs_clean.csv"
    
    # Load data
    df = load_data(input_file)
    
    # Display basic info
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Remove outliers
    df = remove_outliers(df)
    
    # Process job descriptions
    df = process_job_descriptions(df)
    
    # Export cleaned data
    export_cleaned_data(df, output_file)
    
    print("\nData processing completed!")
    print(f"Cleaned dataset saved to: {output_file}")

if __name__ == "__main__":
    main()