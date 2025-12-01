"""
Skill Gap Analysis Module

This module implements comprehensive skill gap analysis for:
1. Profile assessment against market requirements
2. Missing skill identification
3. Personalized recommendation generation
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SkillGapAnalyzer:
    """Analyzes skill gaps and provides recommendations"""
    
    def __init__(self, data_file: str = "data/data_jobs_clean.csv"):
        """
        Initialize skill gap analyzer
        
        Args:
            data_file (str): Path to job data CSV file
        """
        self.data_file = data_file
        self.df = None
        self.market_skills = []
        self.skill_importance = {}
        self.load_data()
        self.extract_market_skills()
        
    def load_data(self):
        """Load job data from CSV file"""
        if os.path.exists(self.data_file):
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.df)} job records for skill analysis")
        else:
            print(f"Data file not found: {self.data_file}")
            
    def extract_market_skills(self):
        """Extract skills from the job market data"""
        if self.df is None:
            return
            
        all_skills = []
        if 'skills_list' in self.df.columns:
            for skills_str in self.df['skills_list'].dropna():
                try:
                    skills_list = json.loads(skills_str) if isinstance(skills_str, str) else skills_str
                    if isinstance(skills_list, list):
                        all_skills.extend([skill.lower().strip() for skill in skills_list])
                except:
                    continue
                    
        # Get unique skills and their frequencies
        skill_counts = Counter(all_skills)
        self.market_skills = list(skill_counts.keys())
        self.skill_importance = {skill: count/len(all_skills) for skill, count in skill_counts.items()}
        
        print(f"Extracted {len(self.market_skills)} unique skills from market data")
        
    def _categorize_skill(self, skill: str) -> str:
        """Categorize a skill into a broad category"""
        skill_lower = skill.lower()
        
        # Programming languages
        if skill_lower in ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'scala', 'r', 'sql']:
            return "Programming Language"
        
        # Data science/ML frameworks
        if any(ml_framework in skill_lower for ml_framework in ['tensorflow', 'pytorch', 'scikit', 'keras', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'xgboost', 'lightgbm']):
            return "Data Science Framework"
        
        # Cloud platforms
        if any(cloud in skill_lower for cloud in ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 'microsoft azure']):
            return "Cloud Platform"
        
        # DevOps/tools
        if any(devops in skill_lower for devops in ['docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'git', 'github', 'gitlab']):
            return "DevOps Tool"
        
        # Databases
        if any(db in skill_lower for db in ['mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sql server', 'cassandra', 'elasticsearch']):
            return "Database"
        
        # Business Intelligence
        if any(bi in skill_lower for bi in ['tableau', 'power bi', 'looker', 'qlik', 'superset']):
            return "Business Intelligence"
        
        # Big Data
        if any(bigdata in skill_lower for bigdata in ['hadoop', 'spark', 'kafka', 'flink', 'hive', 'pig']):
            return "Big Data Technology"
        
        # Web development
        if any(web in skill_lower for web in ['react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'express']):
            return "Web Development"
        
        # Mobile development
        if any(mobile in skill_lower for mobile in ['android', 'ios', 'react native', 'flutter', 'xamarin']):
            return "Mobile Development"
        
        # Default category
        return "Other Skill"
        
    def analyze_profile(self, user_skills: List[str]) -> Dict[str, Any]:
        """
        Analyze a user profile and identify skill gaps
        
        Args:
            user_skills (List[str]): List of user's current skills
            
        Returns:
            Dict: Analysis results including gaps and recommendations
        """
        if not self.market_skills:
            return {
                "profile_skills": user_skills,
                "market_skills": [],
                "missing_skills": [],
                "profile_match": 0.0,
                "recommendations": []
            }
            
        # Normalize user skills
        normalized_user_skills = [skill.lower().strip() for skill in user_skills]
        
        # Identify missing skills
        missing_skills = [skill for skill in self.market_skills if skill not in normalized_user_skills]
        
        # Calculate profile match
        matching_skills = [skill for skill in normalized_user_skills if skill in self.market_skills]
        profile_match = len(matching_skills) / len(self.market_skills) if self.market_skills else 0
        
        # Get skill importance for user's skills
        user_skill_importance = {}
        for skill in normalized_user_skills:
            if skill in self.skill_importance:
                user_skill_importance[skill] = self.skill_importance[skill]
                
        # Recommend top missing skills
        skill_importance_sorted = sorted(
            [(skill, self.skill_importance[skill]) for skill in missing_skills],
            key=lambda x: x[1],
            reverse=True
        )
        
        top_recommendations = skill_importance_sorted[:10]
        recommendations = [
            {
                "skill": skill,
                "importance": importance,
                "market_demand": importance * 100
            }
            for skill, importance in top_recommendations
        ]
        
        # Format missing skills as dictionaries to match Pydantic model
        formatted_missing_skills = [
            {
                "skill": skill,
                "importance": self.skill_importance.get(skill, 0),
                "category": self._categorize_skill(skill)
            }
            for skill in missing_skills[:20]  # Top 20 missing skills
        ]
        
        # Also provide simple list for backward compatibility
        simple_missing_skills = missing_skills[:20]
        
        return {
            "profile_skills": normalized_user_skills,
            "market_skills": self.market_skills[:50],  # Top 50 market skills
            "missing_skills": formatted_missing_skills,
            "missing_skills_simple": simple_missing_skills,  # Backward compatibility
            "profile_match": round(profile_match, 4),
            "skill_importance": user_skill_importance,
            "recommendations": recommendations
        }
        
    def recommend_jobs(self, user_skills: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recommend jobs based on user skills
        
        Args:
            user_skills (List[str]): List of user's current skills
            limit (int): Maximum number of recommendations
            
        Returns:
            List[Dict]: Job recommendations
        """
        if self.df is None:
            return []
            
        normalized_user_skills = [skill.lower().strip() for skill in user_skills]
        recommendations = []
        
        # Score jobs based on skill match
        if 'skills_list' in self.df.columns and 'job_title' in self.df.columns:
            for idx, row in self.df.head(100).iterrows():  # Limit to 100 for performance
                try:
                    job_skills_str = row['skills_list']
                    if job_skills_str:
                        job_skills = json.loads(job_skills_str) if isinstance(job_skills_str, str) else job_skills_str
                        if isinstance(job_skills, list):
                            job_skills = [skill.lower().strip() for skill in job_skills]
                            
                            # Calculate match score
                            matching_skills = [skill for skill in normalized_user_skills if skill in job_skills]
                            match_score = len(matching_skills) / len(job_skills) if job_skills else 0
                            
                            if match_score > 0.3:  # Minimum 30% match
                                recommendations.append({
                                    "job_title": row.get('job_title', 'Unknown Title'),
                                    "company": row.get('company_name', 'Unknown Company'),
                                    "match_score": round(match_score, 4),
                                    "matching_skills": matching_skills,
                                    "total_skills": len(job_skills)
                                })
                except:
                    continue
                    
        # Sort by match score and limit results
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        return recommendations[:limit]
        
    def generate_learning_path(self, missing_skills: List[str], user_level: str = "intermediate") -> List[Dict[str, Any]]:
        """
        Generate personalized learning path for missing skills
        
        Args:
            missing_skills (List[str]): List of skills to learn
            user_level (str): Current skill level (beginner/intermediate/advanced)
            
        Returns:
            List[Dict]: Learning path recommendations
        """
        # Define learning paths based on skill categories
        learning_paths = {
            "python": {
                "beginner": {
                    "course": "Python for Everybody",
                    "provider": "Coursera",
                    "duration": "4-6 months",
                    "skills_covered": ["basics", "data structures", "functions"],
                    "cost": "Free"
                },
                "intermediate": {
                    "course": "Python Data Science Handbook",
                    "provider": "O'Reilly",
                    "duration": "2-3 months",
                    "skills_covered": ["pandas", "numpy", "matplotlib"],
                    "cost": "$40/month"
                },
                "advanced": {
                    "course": "Advanced Python Programming",
                    "provider": "Udemy",
                    "duration": "1-2 months",
                    "skills_covered": ["decorators", "asyncio", "metaclasses"],
                    "cost": "$20"
                }
            },
            "sql": {
                "beginner": {
                    "course": "SQL Basics",
                    "provider": "Khan Academy",
                    "duration": "1-2 months",
                    "skills_covered": ["queries", "joins", "aggregations"],
                    "cost": "Free"
                },
                "intermediate": {
                    "course": "SQL for Data Analysis",
                    "provider": "Udacity",
                    "duration": "2 months",
                    "skills_covered": ["window functions", "CTEs", "optimization"],
                    "cost": "$40/month"
                },
                "advanced": {
                    "course": "Advanced SQL",
                    "provider": "Pluralsight",
                    "duration": "1 month",
                    "skills_covered": ["stored procedures", "triggers", "performance tuning"],
                    "cost": "$30/month"
                }
            },
            "machine learning": {
                "beginner": {
                    "course": "Machine Learning Crash Course",
                    "provider": "Google",
                    "duration": "1 month",
                    "skills_covered": ["linear regression", "classification", "clustering"],
                    "cost": "Free"
                },
                "intermediate": {
                    "course": "Machine Learning Specialization",
                    "provider": "Coursera",
                    "duration": "4-6 months",
                    "skills_covered": ["neural networks", "deep learning", "model evaluation"],
                    "cost": "$49/month"
                },
                "advanced": {
                    "course": "Deep Learning Specialization",
                    "provider": "DeepLearning.AI",
                    "duration": "6 months",
                    "skills_covered": ["CNNs", "RNNs", "transformers"],
                    "cost": "$49/month"
                }
            }
        }
        
        # Generate recommendations for missing skills
        recommendations = []
        for skill in missing_skills[:5]:  # Limit to top 5 skills
            skill_key = skill.lower().strip()
            if skill_key in learning_paths:
                path = learning_paths[skill_key].get(user_level, learning_paths[skill_key]["intermediate"])
                path["skill"] = skill
                recommendations.append(path)
            else:
                # Generic recommendation
                recommendations.append({
                    "skill": skill,
                    "course": f"Master {skill.title()}",
                    "provider": "Udemy/Coursera",
                    "duration": "2-3 months",
                    "skills_covered": [skill],
                    "cost": "$20-50"
                })
                
        return recommendations

def main():
    """Main function to demonstrate skill gap analysis"""
    print("Job Market Skill Gap Analysis")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = SkillGapAnalyzer()
    
    # Example user profile
    user_skills = [
        "python", "sql", "data analysis", "excel", 
        "statistics", "tableau", "communication"
    ]
    
    print(f"\nAnalyzing profile with {len(user_skills)} skills...")
    
    # Analyze profile
    analysis = analyzer.analyze_profile(user_skills)
    print(f"Profile match with market: {analysis['profile_match']:.2%}")
    print(f"Missing skills identified: {len(analysis['missing_skills'])}")
    
    if analysis['recommendations']:
        print("\nTop skill recommendations:")
        for rec in analysis['recommendations'][:5]:
            print(f"  - {rec['skill']} (importance: {rec['market_demand']:.1f}%)")
    
    # Recommend jobs
    job_recommendations = analyzer.recommend_jobs(user_skills)
    print(f"\nJob recommendations: {len(job_recommendations)}")
    if job_recommendations:
        top_job = job_recommendations[0]
        print(f"Best match: {top_job['job_title']} at {top_job['company']} ({top_job['match_score']:.2%})")
    
    # Generate learning path
    if analysis['missing_skills']:
        learning_path = analyzer.generate_learning_path(analysis['missing_skills'][:3])
        print(f"\nLearning path recommendations: {len(learning_path)}")
        if learning_path:
            first_path = learning_path[0]
            print(f"Top recommendation: {first_path['course']} by {first_path['provider']}")

if __name__ == "__main__":
    main()