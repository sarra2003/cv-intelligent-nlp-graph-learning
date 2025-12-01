"""
LLM Integration Module for Job Market Analysis
"""
import os
import json
import requests
import logging
from typing import Dict, List, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM interactions for job market analysis"""
    
    def __init__(self, api_base: str = "http://localhost:11434/api/generate", model: str = "llama2"):
        """
        Initialize LLM manager
        
        Args:
            api_base (str): API base URL (for Ollama compatibility)
            model (str): Model name
        """
        self.api_base = api_base
        self.model = model
        self.session = requests.Session()
        
        # Check for Together AI API key
        self.together_api_key = os.environ.get("TOGETHER_API_KEY", "3e532a18621373aa6dff49a313144f82eebce5839f2995715b408ac3b999ea76")
        self.together_api_base = "https://api.together.xyz/v1"
        
    def is_together_ai_available(self) -> bool:
        """Check if Together AI API key is available and valid"""
        return bool(self.together_api_key and len(self.together_api_key) > 20)
        
    def is_ollama_running(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.session.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def pull_model(self, model_name: Optional[str] = None):
        """
        Pull a model if not available
        
        Args:
            model_name (str): Model name to pull
        """
        model_name = model_name or self.model
        try:
            response = self.session.post(
                "http://localhost:11434/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            if response.status_code == 200:
                logger.info(f"Model {model_name} pulled successfully")
            else:
                logger.error(f"Failed to pull model {model_name}: {response.text}")
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text_together(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using Together AI API
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Generation temperature
            
        Returns:
            str: Generated text
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",  # Using a free serverless model
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.7,
                "top_k": 50,
                "repetition_penalty": 1.1
            }
            
            response = self.session.post(
                f"{self.together_api_base}/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["text"].strip()
            else:
                logger.error(f"Together AI API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating text with Together AI: {e}")
            return ""
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_text(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        Generate text using LLM (prioritizes Together AI, falls back to Ollama)
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate
            temperature (float): Generation temperature
            
        Returns:
            str: Generated text
        """
        # Try Together AI first if API key is available
        if self.is_together_ai_available():
            result = self.generate_text_together(prompt, max_tokens, temperature)
            if result:
                return result
        
        # Fall back to Ollama
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = self.session.post(self.api_base, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
            
    def generate_job_description(self, job_title: str, skills: List[str], company: Optional[str] = None) -> str:
        """
        Generate a job description based on title, skills, and company
        
        Args:
            job_title (str): Job title
            skills (List[str]): Required skills
            company (str): Company name
            
        Returns:
            str: Generated job description
        """
        prompt = f"""
        Generate a professional job description for a {job_title} position.
        
        Required Skills: {', '.join(skills)}
        
        Company: {company if company else 'A leading technology company'}
        
        Include:
        - Job responsibilities
        - Required qualifications
        - Nice-to-have skills
        - Company culture (if company provided)
        
        Keep it concise and professional.
        """
        
        return self.generate_text(prompt, max_tokens=300)
        
    def generate_cv_enhancement(self, current_cv: str, missing_skills: List[str], target_role: str) -> str:
        """
        Generate CV enhancement suggestions
        
        Args:
            current_cv (str): Current CV content
            missing_skills (List[str]): Skills missing for target role
            target_role (str): Target job role
            
        Returns:
            str: Enhanced CV suggestions
        """
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
        
        return self.generate_text(prompt, max_tokens=800)

    def generate_training_recommendations(self, current_skills: List[str], target_skills: List[str]) -> List[Dict[str, Any]]:
        """
        Generate personalized training recommendations
        
        Args:
            current_skills (List[str]): Current skills
            target_skills (List[str]): Target skills to acquire
            
        Returns:
            List[Dict[str, Any]]: Training recommendations
        """
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
        
        response = self.generate_text(prompt, max_tokens=800)
        
        # Try to parse JSON response
        try:
            # Extract JSON from response if it's wrapped in text
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
        except:
            pass
            
        # Return default recommendations if parsing fails
        return [
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
            }
        ]
        
    def answer_market_question(self, question: str, context: str = "") -> str:
        """
        Answer a market-related question with context
        
        Args:
            question (str): Question to answer
            context (str): Additional context
            
        Returns:
            str: Answer to the question
        """
        prompt = f"""
        Answer the following question about the job market:
        
        Question: {question}
        
        Context: {context}
        
        Provide a detailed, data-driven answer with specific examples if possible.
        Keep it professional and informative.
        """
        
        return self.generate_text(prompt, max_tokens=400)
        
    def generate_skill_analysis(self, skills: List[str], market_data: Dict) -> str:
        """
        Generate skill market analysis
        
        Args:
            skills (List[str]): Skills to analyze
            market_data (Dict): Market data context
            
        Returns:
            str: Skill analysis report
        """
        prompt = f"""
        Analyze the following skills in the current job market:
        
        Skills: {', '.join(skills)}
        
        Market Context:
        {json.dumps(market_data, indent=2)}
        
        Provide:
        1. Demand analysis for each skill
        2. Salary implications
        3. Growth trends
        4. Recommendations for skill development
        
        Format as a professional market analysis report.
        """
        
        return self.generate_text(prompt, max_tokens=600)

def main():
    """Main function to demonstrate LLM integration"""
    # Initialize LLM manager
    llm_manager = LLMManager()
    
    # Check if Together AI or Ollama is available
    if llm_manager.is_together_ai_available():
        logger.info("Using Together AI API")
    elif llm_manager.is_ollama_running():
        logger.info("Using Ollama service")
    else:
        logger.warning("Neither Together AI nor Ollama is available.")
        logger.info("For Together AI, set TOGETHER_API_KEY environment variable.")
        logger.info("For Ollama, please start Ollama service.")
        return
    
    try:
        # Example: Generate job description
        job_desc = llm_manager.generate_job_description(
            "Data Scientist",
            ["Python", "Machine Learning", "SQL", "Statistics"],
            "TechCorp"
        )
        logger.info(f"Generated job description: {job_desc[:200]}...")
        
        # Example: Generate training recommendations
        recommendations = llm_manager.generate_training_recommendations(
            ["Python", "SQL"],
            ["Machine Learning", "Deep Learning", "TensorFlow"]
        )
        logger.info(f"Training recommendations: {recommendations}")
        
    except Exception as e:
        logger.error(f"Error in LLM integration: {e}")

if __name__ == "__main__":
    main()