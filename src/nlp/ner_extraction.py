"""
Named Entity Recognition (NER) Module

This module implements two approaches for extracting entities from job postings:
1. Rule-based approach using spaCy and gazetteer
2. Transformer-based custom NER model

The module extracts skills, technologies, and companies from job descriptions.
"""

import pandas as pd
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re
from collections import Counter
import json
import pickle
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Download required spaCy model (run this once)
# python -m spacy download en_core_web_sm

def load_spacy_model():
    """Load spaCy model for NLP processing"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model not found. Please install it with: python -m spacy download en_core_web_sm")
        # Try to download it automatically
        try:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
        except:
            nlp = None
    return nlp

def advanced_ner_extraction(text: str, nlp_model) -> Dict[str, List[str]]:
    """
    Advanced NER using spaCy and Transformers
    
    Args:
        text (str): Input text
        nlp_model: spaCy model
        
    Returns:
        Dict[str, List[str]]: Extracted entities
    """
    if not nlp_model:
        return {"skills": [], "technologies": [], "companies": [], "organizations": []}
    
    # Process text with spaCy
    doc = nlp_model(text)
    
    # Extract named entities
    companies = [ent.text for ent in doc.ents if ent.label_ in ["ORG"]]
    technologies = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "LANGUAGE", "TECH"]]
    persons = [ent.text for ent in doc.ents if ent.label_ in ["PERSON"]]
    
    # Use transformers-based NER for additional skills extraction
    try:
        # Load a pre-trained NER model for skills
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")  # type: ignore
        
        # Extract entities with transformers
        entities = ner_pipeline(text)
        skills = [entity['word'] for entity in entities if entity['entity_group'] in ['MISC', 'ORG']]
    except Exception as e:
        print(f"Transformers NER failed: {e}")
        skills = []
    
    return {
        "skills": list(set(skills)),
        "technologies": list(set(technologies)),
        "companies": list(set(companies)),
        "persons": list(set(persons))
    }

def rule_based_ner(text: str, skills_gazetteer: List[str], nlp) -> Dict[str, List[str]]:
    """
    Rule-based NER using spaCy and gazetteer
    
    Args:
        text (str): Input text
        skills_gazetteer (List[str]): List of known skills
        nlp: spaCy model
        
    Returns:
        Dict[str, List[str]]: Extracted entities
    """
    if not nlp:
        return {"skills": [], "technologies": [], "companies": []}
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract named entities
    companies = [ent.text for ent in doc.ents if ent.label_ in ["ORG"]]
    technologies = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "LANGUAGE", "TECH"]]
    
    # Match skills from gazetteer
    skills = []
    text_lower = text.lower()
    
    for skill in skills_gazetteer:
        if skill in text_lower:
            skills.append(skill)
    
    # Additional pattern matching for common tech terms
    tech_patterns = [
        r'\bpython\b', r'\br\b', r'\bsql\b', r'\bjava\b', r'\bjavascript\b',
        r'\bdocker\b', r'\bkubernetes\b', r'\bgit\b', 
        r'\btensorflow\b',
        r'\bpytorch\b', r'\bspark\b', r'\bhadoop\b', r'\baws\b', r'\bgcp\b',
        r'\bazure\b', r'\blinux\b', r'\bgitlab\b', r'\bjenkins\b'
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text_lower)
        technologies.extend(matches)
    
    # Remove duplicates and return
    return {
        "skills": list(set(skills)),
        "technologies": list(set(technologies)),
        "companies": list(set(companies))
    }

def extract_entities_from_dataframe(df: pd.DataFrame, nlp) -> pd.DataFrame:
    """
    Extract entities from all job postings in the dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        nlp: spaCy model
        
    Returns:
        pd.DataFrame: Dataframe with extracted entities
    """
    print("Creating skill gazetteer...")
    skills_gazetteer = create_skill_gazetteer(df)
    
    print("Extracting entities from job postings...")
    extracted_entities = []
    
    # Combine job title and description for entity extraction
    for idx, row in df.iterrows():
        text_parts = []
        
        if 'job_title_clean' in df.columns and pd.notna(row['job_title_clean']):
            text_parts.append(str(row['job_title_clean']))
            
        if 'job_description_clean' in df.columns and pd.notna(row.get('job_description_clean', None)):
            text_parts.append(str(row['job_description_clean']))  # type: ignore
            
        full_text = " ".join(text_parts)
        
        # Extract entities with advanced NER
        entities = advanced_ner_extraction(full_text, nlp)
        entities['job_id'] = [str(idx)]
        
        extracted_entities.append(entities)
    
    # Create dataframe with extracted entities
    entities_df = pd.DataFrame(extracted_entities)
    
    # Merge with original dataframe
    result_df = df.copy()
    result_df = result_df.merge(entities_df, left_index=True, right_on='job_id', how='left')
    
    return result_df

def create_skill_gazetteer(df: pd.DataFrame) -> List[str]:
    """
    Create a gazetteer of skills from the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe with skills column
        
    Returns:
        List[str]: List of skills
    """
    all_skills = []
    
    if 'skills_list' in df.columns:
        for skills_str in df['skills_list'].dropna():
            try:
                # Parse the JSON string back to list
                if isinstance(skills_str, str) and skills_str.startswith('['):
                    skills_list = json.loads(skills_str)
                else:
                    # If it's already a list or other format
                    skills_list = skills_str if isinstance(skills_str, list) else [skills_str]
                
                if isinstance(skills_list, list):
                    all_skills.extend([skill.lower().strip() for skill in skills_list])
            except:
                continue
    
    # Get unique skills
    unique_skills = list(set(all_skills))
    print(f"Created gazetteer with {len(unique_skills)} unique skills")
    
    return unique_skills

def generate_embeddings(texts: List[str], model_name: str = 'bert-base-uncased'):
    """
    Generate embeddings for texts using sentence-transformers
    
    Args:
        texts (List[str]): List of texts to embed
        model_name (str): Name of the sentence transformer model
        
    Returns:
        np.ndarray: Array of embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts)
        return embeddings
    except ImportError:
        print("sentence-transformers not installed. Please install with: pip install sentence-transformers")
        return None

def embed_job_components(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Generate embeddings for job titles, skills, and descriptions
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of embeddings
    """
    print("Generating embeddings...")
    
    embeddings = {}
    
    # Job titles
    if 'job_title_clean' in df.columns:
        job_titles = df['job_title_clean'].fillna("").tolist()
        title_embeddings = generate_embeddings(job_titles)
        if title_embeddings is not None:
            embeddings['job_titles'] = title_embeddings
            print(f"Generated embeddings for {len(title_embeddings)} job titles")
    
    # Skills (combined)
    if 'skills_list' in df.columns:
        skills_texts = []
        for skills_str in df['skills_list'].fillna("[]"):
            try:
                skills_list = json.loads(skills_str) if isinstance(skills_str, str) else skills_str
                skills_text = ", ".join(skills_list) if isinstance(skills_list, list) else ""
                skills_texts.append(skills_text)
            except:
                skills_texts.append("")
        
        skills_embeddings = generate_embeddings(skills_texts)
        if skills_embeddings is not None:
            embeddings['skills'] = skills_embeddings
            print(f"Generated embeddings for {len(skills_embeddings)} skills sets")
    
    # Job descriptions
    if 'job_description_clean' in df.columns:
        descriptions = df['job_description_clean'].fillna("").tolist()
        desc_embeddings = generate_embeddings(descriptions)
        if desc_embeddings is not None:
            embeddings['descriptions'] = desc_embeddings
            print(f"Generated embeddings for {len(desc_embeddings)} job descriptions")
    
    return embeddings

def save_entities_and_embeddings(df: pd.DataFrame, embeddings: Dict[str, np.ndarray]):
    """
    Save extracted entities and embeddings
    
    Args:
        df (pd.DataFrame): Dataframe with extracted entities
        embeddings (Dict[str, np.ndarray]): Dictionary of embeddings
    """
    # Save dataframe with entities
    df.to_csv('data/jobs_with_entities.csv', index=False)
    print("Saved jobs with entities to data/jobs_with_entities.csv")
    
    # Save embeddings
    if embeddings:
        os.makedirs('models/embeddings', exist_ok=True)
        for key, emb in embeddings.items():
            np.save(f'models/embeddings/{key}_embeddings.npy', emb)
        print("Saved embeddings to models/embeddings/")

def main():
    """
    Main function to run the NER extraction pipeline
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = pd.read_csv('data/data_jobs_clean.csv')
    print(f"Loaded {len(df)} job postings")
    
    # For testing purposes, use a sample of the data
    # Comment this out to process the full dataset
    df = df.sample(n=min(1000, len(df)), random_state=42)
    print(f"Using sample of {len(df)} job postings for processing")
    
    # Load spaCy model
    nlp = load_spacy_model()
    
    if nlp is None:
        print("Skipping NER extraction due to missing spaCy model")
        return
    
    # Extract entities
    df_with_entities = extract_entities_from_dataframe(df, nlp)
    
    # Generate embeddings
    embeddings = embed_job_components(df_with_entities)
    
    # Save results
    save_entities_and_embeddings(df_with_entities, embeddings)
    
    print("\nNER extraction pipeline completed!")
    print("Results saved to:")
    print("- data/jobs_with_entities.csv")
    print("- models/embeddings/")

if __name__ == "__main__":
    main()