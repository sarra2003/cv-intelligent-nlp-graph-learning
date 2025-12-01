"""
CV Generation Module

This module implements automated CV enhancement and recommendation generation:
1. CV content analysis
2. Skill gap identification in CV
3. Automated CV enhancement suggestions
4. Personalized training recommendations
"""

import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import LLM manager
try:
    from src.llm_integration import LLMManager
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMManager = None  # Define fallback
    print("LLM integration not available. Using template-based generation.")

class CVGenerator:
    """Generates enhanced CVs and career recommendations"""
    
    def __init__(self, data_file: str = "data/data_jobs_clean.csv"):
        """
        Initialize CV generator
        
        Args:
            data_file (str): Path to job data CSV file
        """
        self.data_file = data_file
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load job data from CSV file"""
        if os.path.exists(self.data_file):
            self.df = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.df)} job records for CV generation")
        else:
            print(f"Data file not found: {self.data_file}")
            
    def analyze_cv(self, cv_text: str) -> Dict[str, Any]:
        """
        Analyze CV content and extract key information
        
        Args:
            cv_text (str): Full CV text content
            
        Returns:
            Dict: Analysis results
        """
        # Simple keyword-based analysis for demo
        cv_lower = cv_text.lower()
        
        # Extract skills from CV
        tech_keywords = [
            'python', 'sql', 'java', 'javascript', 'r', 'scala', 'matlab',
            'tensorflow', 'pytorch', 'scikit-learn', 'keras',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'tableau', 'power bi', 'matplotlib', 'seaborn',
            'git', 'jenkins', 'ci/cd', 'agile', 'scrum'
        ]
        
        found_skills = [skill for skill in tech_keywords if skill in cv_lower]
        
        # Extract experience indicators
        experience_indicators = [
            'experience', 'years', 'worked', 'developed', 'implemented',
            'managed', 'led', 'created', 'built', 'designed'
        ]
        
        experience_mentions = [word for word in experience_indicators if word in cv_lower]
        
        return {
            "extracted_skills": found_skills,
            "experience_indicators": experience_mentions,
            "cv_length": len(cv_text.split()),
            "sections_identified": self._identify_cv_sections(cv_text)
        }
        
    def _identify_cv_sections(self, cv_text: str) -> List[str]:
        """Identify common CV sections"""
        section_keywords = [
            'summary', 'objective', 'experience', 'education', 
            'skills', 'projects', 'certifications', 'awards'
        ]
        
        found_sections = []
        cv_lower = cv_text.lower()
        
        for section in section_keywords:
            if section in cv_lower:
                found_sections.append(section.title())
                
        return found_sections
        
    def generate_enhanced_cv(self, current_cv: str, missing_skills: List[str], 
                           target_role: str, market_data: Optional[Dict] = None) -> str:
        """
        Generate enhanced CV content with improvements
        
        Args:
            current_cv (str): Current CV content
            missing_skills (List[str]): Skills to incorporate
            target_role (str): Target job role
            market_data (Dict): Market context data
            
        Returns:
            str: Enhanced CV content
        """
        if LLM_AVAILABLE and LLMManager:
            # Use LLM for dynamic generation
            llm_manager = LLMManager()
            if llm_manager.is_together_ai_available():
                try:
                    prompt = f"""
                    Enhance the following CV for a {target_role} position.
                    
                    Current CV:
                    {current_cv}
                    
                    Missing Skills to Address:
                    {', '.join(missing_skills)}
                    
                    Provide specific suggestions to:
                    1. Optimize the CV structure
                    2. Highlight relevant experience
                    3. Incorporate missing skills naturally
                    4. Add impactful action verbs
                    5. Quantify achievements where possible
                    
                    Format as a professional CV enhancement guide.
                    """
                    
                    return llm_manager.generate_text(prompt, max_tokens=800)
                except Exception as e:
                    print(f"LLM generation failed: {e}")
        
        # Fallback to template-based generation
        return self._generate_template_cv(current_cv, missing_skills, target_role)
        
    def _generate_template_cv(self, current_cv: str, missing_skills: List[str], 
                            target_role: str) -> str:
        """Generate CV enhancement using templates"""
        enhancement_guide = f"""
CV ENHANCEMENT GUIDE FOR {target_role.upper()} POSITION
=================================================

CURRENT CV ANALYSIS:
-------------------
Your CV contains solid foundational elements. Based on market analysis, here are key areas for improvement:

1. SKILL INTEGRATION
   Missing skills to incorporate:
   {', '.join(missing_skills[:5])}
   
   Suggested integrations:
   - Add specific projects demonstrating {missing_skills[0] if missing_skills else 'technical skills'}
   - Include quantifiable achievements with relevant technologies
   - Mention {missing_skills[1] if len(missing_skills) > 1 else 'key skills'} in experience descriptions

2. EXPERIENCE OPTIMIZATION
   Recommended action verbs:
   - "Developed" → "Engineered scalable solutions for..."
   - "Managed" → "Led cross-functional team to deliver..."
   - "Worked on" → "Architected and implemented..."

3. STRUCTURE IMPROVEMENTS
   Suggested CV structure:
   - Professional Summary (2-3 lines highlighting key qualifications)
   - Core Competencies (technical skills section)
   - Professional Experience (reverse chronological with achievements)
   - Education & Certifications
   - Key Projects (2-3 relevant projects with impact metrics)

4. TARGETED CONTENT
   For {target_role} roles, emphasize:
   - Data-driven decision making
   - Cross-functional collaboration
   - Technical problem-solving
   - Business impact of your work

SAMPLE ENHANCED EXPERIENCE ENTRY:
--------------------------------
Senior Data Analyst | TechCorp | Jan 2020 - Present
• Engineered automated reporting system using Python and SQL, reducing manual reporting time by 70%
• Led data analysis for customer segmentation project, resulting in 15% increase in targeted marketing ROI
• Collaborated with engineering teams to implement {missing_skills[0] if missing_skills else 'machine learning'} models for predictive analytics

RECOMMENDED CERTIFICATIONS:
--------------------------
• {target_role} Professional Certification
• Cloud Platform Certification (AWS/Azure/GCP)
• Advanced Analytics Certification

This enhanced CV positions you as a strong candidate for {target_role} positions by highlighting relevant skills and quantifiable achievements.
        """
        
        return enhancement_guide.strip()
        
    def generate_training_recommendations(self, current_skills: List[str], 
                                        target_skills: List[str]) -> List[Dict[str, Any]]:
        """
        Generate personalized training recommendations
        
        Args:
            current_skills (List[str]): Current skill set
            target_skills (List[str]): Target skill set
            
        Returns:
            List[Dict]: Training recommendations
        """
        if LLM_AVAILABLE and LLMManager:
            # Use LLM for dynamic generation
            llm_manager = LLMManager()
            if llm_manager.is_together_ai_available():
                try:
                    prompt = f"""
                    Generate personalized training recommendations for someone with skills: {', '.join(current_skills)}
                    who wants to acquire: {', '.join(target_skills)}
                    
                    Provide 5 specific recommendations including:
                    1. Course/learning path name
                    2. Platform/provider (e.g., Coursera, Udemy, edX)
                    3. Estimated duration
                    4. Key skills covered
                    5. Difficulty level (Beginner/Intermediate/Advanced)
                    6. Cost range if applicable
                    
                    Format as a JSON array of objects with these fields.
                    """
                    
                    response = llm_manager.generate_text(prompt, max_tokens=800)
                    
                    # Try to parse JSON response
                    try:
                        import re
                        json_match = re.search(r'\[.*\]', response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            return json.loads(json_str)
                    except:
                        pass
                except Exception as e:
                    print(f"LLM generation failed: {e}")
        
        # Fallback to template-based recommendations
        return self._generate_template_recommendations(current_skills, target_skills)
        
    def _generate_template_recommendations(self, current_skills: List[str], 
                                         target_skills: List[str]) -> List[Dict[str, Any]]:
        """Generate training recommendations using templates"""
        recommendations = [
            {
                "course": "Data Science Fundamentals",
                "provider": "Coursera",
                "duration": "3-4 months",
                "skills_covered": ["Python", "Statistics", "Machine Learning"],
                "difficulty": "Intermediate",
                "cost": "$49/month"
            },
            {
                "course": "Cloud Computing Specialization",
                "provider": "AWS/Azure",
                "duration": "2-3 months",
                "skills_covered": ["Cloud Architecture", "DevOps", "Security"],
                "difficulty": "Intermediate",
                "cost": "$100-200"
            },
            {
                "course": "Advanced Machine Learning",
                "provider": "edX",
                "duration": "4-6 months",
                "skills_covered": ["Deep Learning", "Neural Networks", "NLP"],
                "difficulty": "Advanced",
                "cost": "$300-500"
            },
            {
                "course": "Data Visualization & Storytelling",
                "provider": "Udemy",
                "duration": "1-2 months",
                "skills_covered": ["Tableau", "Power BI", "Communication"],
                "difficulty": "Beginner",
                "cost": "$20-50"
            },
            {
                "course": "SQL for Data Analysis",
                "provider": "Khan Academy",
                "duration": "1-2 months",
                "skills_covered": ["Advanced SQL", "Database Design", "Optimization"],
                "difficulty": "Intermediate",
                "cost": "Free"
            }
        ]
        
        return recommendations

def main():
    """Main function to demonstrate CV generation"""
    print("Automated CV Enhancement System")
    print("=" * 40)
    
    # Initialize generator
    generator = CVGenerator()
    
    # Example CV
    sample_cv = """
    John Doe
    Data Analyst
    
    Experience:
    - Worked on data analysis projects
    - Used Python and SQL for data processing
    - Created reports for management
    
    Skills:
    - Python
    - SQL
    - Excel
    - Statistics
    """
    
    # Analyze CV
    print("Analyzing CV...")
    analysis = generator.analyze_cv(sample_cv)
    print(f"Extracted skills: {len(analysis['extracted_skills'])}")
    print(f"CV sections: {', '.join(analysis['sections_identified'])}")
    
    # Generate enhanced CV
    missing_skills = ["Machine Learning", "Cloud Computing", "Data Visualization"]
    target_role = "Senior Data Scientist"
    
    print(f"\nGenerating enhancement for {target_role}...")
    enhanced_cv = generator.generate_enhanced_cv(sample_cv, missing_skills, target_role)
    print("Enhanced CV generated successfully!")
    
    # Generate training recommendations
    current_skills = ["Python", "SQL", "Statistics"]
    target_skills = ["Machine Learning", "Deep Learning", "Cloud Platforms"]
    
    print("\nGenerating training recommendations...")
    recommendations = generator.generate_training_recommendations(current_skills, target_skills)
    print(f"Generated {len(recommendations)} recommendations")
    
    if recommendations:
        top_rec = recommendations[0]
        print(f"Top recommendation: {top_rec['course']} by {top_rec['provider']}")

if __name__ == "__main__":
    main()