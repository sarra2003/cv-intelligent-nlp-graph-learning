"""
CV Profiling Module

This module analyzes CVs to:
1. Extract skills from PDF CVs
2. Compare with target job requirements
3. Identify missing skills
4. Generate improved CV suggestions
"""

import pandas as pd
import numpy as np
import json
import re
import os
from typing import List, Dict, Set
import warnings
warnings.filterwarnings('ignore')

# Try to import PDF processing libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 not installed. Please install with: pip install PyPDF2")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("pdfplumber not installed. Please install with: pip install pdfplumber")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from PDF file
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        str: Extracted text
    """
    text = ""
    
    # Try PyPDF2 first
    if PYPDF2_AVAILABLE:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"PyPDF2 extraction failed: {e}")
    
    # Try pdfplumber as fallback
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")
    
    return text

def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract skills from CV text using pattern matching
    
    Args:
        text (str): CV text
        
    Returns:
        List[str]: Extracted skills
    """
    # Common data science skills and technologies
    common_skills = [
        # Programming languages
        'python', 'r', 'sql', 'java', 'javascript', 'scala', 'c++', 'matlab',
        
        # Data science libraries
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'plotly',
        'tensorflow', 'pytorch', 'keras', 'xgboost', 'lightgbm',
        
        # Big data technologies
        'spark', 'hadoop', 'hive', 'kafka', 'flink', 'airflow',
        
        # Cloud platforms
        'aws', 'azure', 'gcp', 'google cloud', 'amazon web services',
        
        # Databases
        'postgresql', 'mysql', 'mongodb', 'cassandra', 'redis', 'elasticsearch',
        
        # DevOps/tools
        'docker', 'kubernetes', 'git', 'jenkins', 'linux', 'bash',
        
        # ML concepts
        'machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision',
        'data mining', 'statistical analysis', 'a/b testing', 'feature engineering',
        
        # Data visualization
        'tableau', 'power bi', 'qlik', 'd3.js',
        
        # Other relevant skills
        'data engineering', 'data analysis', 'data modeling', 'etl', 'api',
        'rest', 'agile', 'scrum', 'jira'
    ]
    
    # Normalize text
    text_lower = text.lower()
    
    # Extract skills
    found_skills = []
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill)
    
    # Additional pattern matching for skills not in predefined list
    # This is a simple example - in practice, you'd use more sophisticated NER
    tech_patterns = [
        r'\b\d+ years of experience\b',
        r'\b\d+ years working with\b',
        r'\bproficient in \w+\b',
        r'\bfamiliar with \w+\b',
        r'\bexperience with \w+\b'
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Extract the skill name from the pattern
            words = match.split()
            if 'proficient' in match or 'familiar' in match or 'experience' in match:
                skill_candidate = words[-1] if words else ''
                if skill_candidate and len(skill_candidate) > 2:
                    found_skills.append(skill_candidate)
    
    return list(set(found_skills))  # Remove duplicates

def load_job_requirements(job_id: str = None) -> Dict:
    """
    Load target job requirements
    
    Args:
        job_id (str): Specific job ID to load (optional)
        
    Returns:
        Dict: Job requirements
    """
    # For demo, we'll create a sample job profile
    # In practice, you would load from your job database
    
    sample_requirements = {
        "job_title": "Senior Data Scientist",
        "required_skills": [
            "python", "machine learning", "deep learning", "tensorflow", "pytorch",
            "pandas", "numpy", "scikit-learn", "sql", "statistics",
            "data visualization", "experimentation", "communication"
        ],
        "preferred_skills": [
            "aws", "spark", "docker", "kubernetes", "nlp", "computer vision"
        ],
        "experience_years": 5
    }
    
    return sample_requirements

def compare_cv_to_job(cv_skills: List[str], job_requirements: Dict) -> Dict:
    """
    Compare CV skills to job requirements
    
    Args:
        cv_skills (List[str]): Skills extracted from CV
        job_requirements (Dict): Job requirements
        
    Returns:
        Dict: Comparison results
    """
    required_skills = set(job_requirements.get('required_skills', []))
    preferred_skills = set(job_requirements.get('preferred_skills', []))
    cv_skills_set = set(cv_skills)
    
    # Find missing skills
    missing_required = required_skills - cv_skills_set
    missing_preferred = preferred_skills - cv_skills_set
    
    # Find matching skills
    matching_required = required_skills & cv_skills_set
    matching_preferred = preferred_skills & cv_skills_set
    
    # Calculate match percentages
    required_match_pct = len(matching_required) / len(required_skills) * 100 if required_skills else 0
    preferred_match_pct = len(matching_preferred) / len(preferred_skills) * 100 if preferred_skills else 0
    
    return {
        "matching_required_skills": list(matching_required),
        "missing_required_skills": list(missing_required),
        "matching_preferred_skills": list(matching_preferred),
        "missing_preferred_skills": list(missing_preferred),
        "required_skills_match_percentage": required_match_pct,
        "preferred_skills_match_percentage": preferred_match_pct,
        "overall_match_score": (required_match_pct * 0.8 + preferred_match_pct * 0.2)  # Weighted score
    }

def generate_cv_improvement_suggestions(comparison_results: Dict, job_requirements: Dict) -> List[str]:
    """
    Generate suggestions for CV improvement
    
    Args:
        comparison_results (Dict): Results from CV-job comparison
        job_requirements (Dict): Job requirements
        
    Returns:
        List[str]: Improvement suggestions
    """
    suggestions = []
    
    # Suggest adding missing required skills
    missing_required = comparison_results.get('missing_required_skills', [])
    if missing_required:
        suggestions.append(f"Add these required skills to your CV: {', '.join(missing_required)}")
    
    # Suggest adding missing preferred skills
    missing_preferred = comparison_results.get('missing_preferred_skills', [])
    if missing_preferred:
        suggestions.append(f"Consider adding these preferred skills: {', '.join(missing_preferred)}")
    
    # General suggestions based on match score
    overall_score = comparison_results.get('overall_match_score', 0)
    
    if overall_score < 50:
        suggestions.append("Your CV needs significant improvement to match this role. Focus on gaining relevant experience.")
    elif overall_score < 75:
        suggestions.append("Your CV is on the right track but needs some enhancement to be competitive.")
    else:
        suggestions.append("Your CV is well-aligned with this role. Consider highlighting your achievements more prominently.")
    
    # Specific suggestions
    suggestions.append("Quantify your achievements with specific metrics and results.")
    suggestions.append("Include relevant projects that demonstrate your skills.")
    suggestions.append("Tailor your CV summary to match the job description keywords.")
    
    return suggestions

def generate_improved_cv_template(cv_text: str, comparison_results: Dict, suggestions: List[str]) -> str:
    """
    Generate an improved CV template (simulated)
    
    Args:
        cv_text (str): Original CV text
        comparison_results (Dict): Comparison results
        suggestions (List[str]): Improvement suggestions
        
    Returns:
        str: Improved CV template
    """
    # This is a simulated improvement - in practice, you would use an LLM
    improved_cv = f"""
IMPROVED CV TEMPLATE

[Your Name]
[Contact Information]

PROFESSIONAL SUMMARY
Experienced data professional with skills in {', '.join(comparison_results.get('matching_required_skills', [])[:3])}.
Looking to leverage expertise in data science and machine learning.

KEY SKILLS
{chr(10).join(['• ' + skill for skill in comparison_results.get('matching_required_skills', [])])}
{chr(10).join(['• ' + skill for skill in comparison_results.get('matching_preferred_skills', [])])}

PROFESSIONAL EXPERIENCE
[Include detailed descriptions with quantified achievements]

EDUCATION
[Your educational background]

CERTIFICATIONS
[Relevant certifications]

PROJECTS
[Data science projects demonstrating your skills]

RECOMMENDATIONS FOR IMPROVEMENT:
{chr(10).join(['• ' + suggestion for suggestion in suggestions])}
"""
    
    return improved_cv

def analyze_cv(cv_path: str, job_requirements: Dict = None) -> Dict:
    """
    Complete CV analysis pipeline
    
    Args:
        cv_path (str): Path to CV PDF
        job_requirements (Dict): Job requirements (optional)
        
    Returns:
        Dict: Analysis results
    """
    print(f"Analyzing CV: {cv_path}")
    
    # Extract text from PDF
    cv_text = extract_text_from_pdf(cv_path)
    if not cv_text:
        return {"error": "Failed to extract text from CV"}
    
    print(f"Extracted {len(cv_text)} characters from CV")
    
    # Extract skills from CV
    cv_skills = extract_skills_from_text(cv_text)
    print(f"Extracted {len(cv_skills)} skills from CV: {cv_skills}")
    
    # Load job requirements
    if job_requirements is None:
        job_requirements = load_job_requirements()
    
    # Compare CV to job requirements
    comparison_results = compare_cv_to_job(cv_skills, job_requirements)
    
    # Generate improvement suggestions
    suggestions = generate_cv_improvement_suggestions(comparison_results, job_requirements)
    
    # Generate improved CV template
    improved_cv = generate_improved_cv_template(cv_text, comparison_results, suggestions)
    
    # Prepare results
    results = {
        "cv_skills": cv_skills,
        "job_requirements": job_requirements,
        "comparison_results": comparison_results,
        "improvement_suggestions": suggestions,
        "improved_cv_template": improved_cv
    }
    
    return results

def main():
    """
    Main function to demonstrate CV profiling
    """
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # For demo, we'll create a sample CV text file
    sample_cv_text = """
    John Doe
    Data Scientist
    
    EXPERIENCE
    
    Senior Data Scientist, TechCorp (2020-Present)
    • Developed machine learning models using Python and scikit-learn
    • Worked with large datasets using pandas and numpy
    • Created data visualizations with matplotlib and seaborn
    • Experience with SQL databases and data warehousing
    • Proficient in statistical analysis and hypothesis testing
    
    Data Analyst, DataSystems (2018-2020)
    • Analyzed business metrics and created reports
    • Used R for statistical modeling
    • Built dashboards with Tableau
    • Collaborated with cross-functional teams
    
    SKILLS
    • Python, R, SQL
    • Machine Learning, Statistics
    • Pandas, NumPy, Scikit-learn
    • Tableau, Matplotlib
    • Git, Linux
    
    EDUCATION
    M.S. in Data Science, University of California (2018)
    B.S. in Mathematics, Stanford University (2016)
    """
    
    # Save sample CV
    sample_cv_path = "data/sample_cv.txt"
    with open(sample_cv_path, 'w') as f:
        f.write(sample_cv_text)
    
    # Analyze sample CV
    results = analyze_cv(sample_cv_path)
    
    # Save results
    with open('outputs/cv_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save improved CV template
    if 'improved_cv_template' in results:
        with open('outputs/improved_cv_template.txt', 'w') as f:
            f.write(results['improved_cv_template'])
    
    print("\nCV profiling completed!")
    print("Results saved to:")
    print("- outputs/cv_analysis_results.json")
    print("- outputs/improved_cv_template.txt")
    
    # Print summary
    if 'comparison_results' in results:
        comp = results['comparison_results']
        print(f"\nCV Analysis Summary:")
        print(f"Required Skills Match: {comp.get('required_skills_match_percentage', 0):.1f}%")
        print(f"Preferred Skills Match: {comp.get('preferred_skills_match_percentage', 0):.1f}%")
        print(f"Overall Match Score: {comp.get('overall_match_score', 0):.1f}%")

if __name__ == "__main__":
    main()