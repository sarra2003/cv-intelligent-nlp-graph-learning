"""
Job Market Analysis API
Provides data-driven insights without chatbot functionality
"""

import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import base64
import logging
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Import our analysis modules
from src.skill_gap_analysis import SkillGapAnalyzer
from src.time_series_analysis import TimeSeriesAnalyzer
from src.cv_generation import CVGenerator

# Add new import for OCR processing
from src.ocr_cv_processing import OCRProcessor

# Import Graph-RAG module
from src.rag.graph_rag import GraphRAG

# Initialize FastAPI app
app = FastAPI(title="Job Market Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    message: str
    total_jobs: int

class GraphDataResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

class JobStatsResponse(BaseModel):
    total_jobs: int
    top_skills: List[Dict[str, Any]]
    average_salary: float
    job_categories: List[Dict[str, Any]]

class SkillAnalysisRequest(BaseModel):
    profile_skills: List[str]

class SkillAnalysisResponse(BaseModel):
    profile_skills: List[str]
    missing_skills: List[Dict[str, Any]]
    missing_skills_simple: Optional[List[str]] = None
    profile_match: float
    recommendations: List[Dict[str, Any]]

class JobRecommendationRequest(BaseModel):
    user_skills: List[str]
    limit: int = 10

class JobRecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

class SalaryTrendResponse(BaseModel):
    dates: List[str]
    values: List[float]
    metric: str

class TechnologyTrendResponse(BaseModel):
    trends: List[Dict[str, Any]]

class CVAnalysisRequest(BaseModel):
    cv_text: str
    target_role: str = "Data Scientist"
    include_market_analysis: bool = True

class EnhancedCVResponse(BaseModel):
    extracted_text: str
    cv_analysis: Dict[str, Any]
    ats_score: Dict[str, Any]  # Changed from float to Dict to accommodate full ATS result
    enhanced_cv: str
    latex_cv: str
    market_trends: Optional[Dict[str, Any]]
    skill_recommendations: Optional[List[Dict[str, Any]]]

class CVImageRequest(BaseModel):
    image_data: str  # Changed from image_base64 to match frontend
    target_role: str = "Data Scientist"
    include_trends: bool = True
    include_market_analysis: bool = True

class JobPredictionRequest(BaseModel):
    job_title: str
    prediction_type: str  # salary, growth, demand, skills

class JobPredictionResponse(BaseModel):
    job_title: str
    prediction_type: str
    prediction: Dict[str, Any]
    confidence: float

class PredictionResponse(BaseModel):
    predictions: Dict[str, Any]

# RAG Models
class RAGQueryRequest(BaseModel):
    question: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float

# Initialize analyzers
skill_analyzer = SkillGapAnalyzer()
time_analyzer = TimeSeriesAnalyzer()
cv_generator = CVGenerator()
ocr_processor = OCRProcessor()

# Initialize Graph-RAG system
rag_system = GraphRAG()
rag_system.build_system()

def load_data_file(filename: str) -> pd.DataFrame:
    """Load data file with error handling"""
    try:
        file_path = os.path.join("data", filename)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            # Create mock data for demo
            return pd.DataFrame({
                'job_title': ['Data Scientist', 'Machine Learning Engineer', 'Data Analyst'],
                'skills_list': ['["python", "sql", "machine learning"]', '["python", "tensorflow", "aws"]', '["sql", "excel", "tableau"]'],
                'salary': [120000, 140000, 80000]
            })
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# Health check endpoint
@app.get("/api/health")
async def health_check() -> HealthResponse:
    """Health check endpoint"""
    df = load_data_file("data_jobs_clean.csv")
    total_jobs = len(df) if not df.empty else 0
    
    return HealthResponse(
        status="healthy",
        message="Job Market Analysis API is running",
        total_jobs=total_jobs
    )

# Job statistics endpoint - fix the job_categories structure
@app.get("/api/jobs/stats")
async def get_job_statistics() -> JobStatsResponse:
    """Get job market statistics"""
    df = load_data_file("data_jobs_clean.csv")
    
    if df.empty:
        return JobStatsResponse(
            total_jobs=0,
            top_skills=[],
            average_salary=0.0,
            job_categories=[]
        )
    
    # Get top skills
    all_skills = []
    for skills_list in df['skills_list'].dropna():
        try:
            skills = json.loads(skills_list)
            all_skills.extend(skills)
        except:
            continue
    
    from collections import Counter
    skill_counts = Counter(all_skills)
    top_skills = [{"skill": skill, "count": count} for skill, count in skill_counts.most_common(10)]
    
    # Calculate average salary
    avg_salary = float(df['salary'].mean()) if 'salary' in df.columns and not df['salary'].empty else 0.0
    
    # Get job categories
    job_categories = []
    if 'job_title' in df.columns:
        category_counts = df['job_title'].value_counts().head(10)
        job_categories = [{"category": cat, "count": int(count)} for cat, count in category_counts.items()]
    
    return JobStatsResponse(
        total_jobs=len(df),
        top_skills=top_skills,
        average_salary=round(avg_salary, 2),
        job_categories=job_categories
    )

# Graph data endpoint
@app.get("/api/graph/data")
async def get_graph_data() -> GraphDataResponse:
    """Get graph data for visualization"""
    try:
        # Load the graph file if it exists
        graph_file = "models/job_graph.pkl"
        if os.path.exists(graph_file):
            import networkx as nx
            import pickle
            
            # Use pickle to load the graph
            with open(graph_file, 'rb') as f:
                G = pickle.load(f)
            
            # Get all nodes and edges
            nodes = []
            edges = []
            
            # Separate nodes by type to ensure balanced representation
            job_nodes = []
            company_nodes = []
            skill_nodes = []
            
            for node, data in G.nodes(data=True):
                node_type = data.get('node_type', 'unknown')
                node_data = {"id": str(node), "label": str(node)}
                node_data.update(data)
                
                if node_type == 'job':
                    job_nodes.append(node_data)
                elif node_type == 'company':
                    company_nodes.append(node_data)
                elif node_type == 'skill':
                    skill_nodes.append(node_data)
                else:
                    nodes.append(node_data)
            
            # Balance the representation - take up to 50 of each type
            nodes.extend(job_nodes[:50])
            nodes.extend(company_nodes[:50])
            nodes.extend(skill_nodes[:50])
            
            # Get edges for these nodes only
            selected_node_ids = {node["id"] for node in nodes}
            for source, target in G.edges():
                if source in selected_node_ids and target in selected_node_ids:
                    edges.append({
                        "source": str(source),
                        "target": str(target)
                    })
            
            return GraphDataResponse(nodes=nodes, edges=edges)
        else:
            # Return mock data if graph file doesn't exist
            mock_nodes = [
                {"id": "job1", "label": "Data Scientist", "type": "job"},
                {"id": "job2", "label": "Machine Learning Engineer", "type": "job"},
                {"id": "comp1", "label": "Google", "type": "company"},
                {"id": "comp2", "label": "Microsoft", "type": "company"},
                {"id": "skill1", "label": "Python", "type": "skill"},
                {"id": "skill2", "label": "Machine Learning", "type": "skill"}
            ]
            
            mock_edges = [
                {"source": "comp1", "target": "job1"},
                {"source": "comp2", "target": "job2"},
                {"source": "job1", "target": "skill1"},
                {"source": "job1", "target": "skill2"},
                {"source": "job2", "target": "skill1"},
                {"source": "job2", "target": "skill2"}
            ]
            
            return GraphDataResponse(nodes=mock_nodes, edges=mock_edges)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading graph data: {str(e)}")

# Skill gap analysis endpoint - updated to match frontend expectations
@app.post("/api/analysis/skills")
async def analyze_skills(request: SkillAnalysisRequest) -> SkillAnalysisResponse:
    """Analyze user skills against market requirements"""
    try:
        # Extract skills from the profile text
        user_skills = request.profile_skills if isinstance(request.profile_skills, list) else [request.profile_skills]
        
        # Clean and process skills
        extracted_skills = []
        for skill in user_skills:
            if isinstance(skill, str):
                # Split by common delimiters and clean
                skill_parts = [s.strip().lower() for s in skill.split(',')]
                extracted_skills.extend([s for s in skill_parts if s])
            else:
                extracted_skills.append(str(skill).lower())
        
        # Remove duplicates while preserving order
        seen = set()
        extracted_skills = [s for s in extracted_skills if not (s in seen or seen.add(s))]
        
        # Handle case where no skills were extracted
        if not extracted_skills:
            extracted_skills = [str(skill) for skill in user_skills if skill is not None]
            
        analysis = skill_analyzer.analyze_profile(extracted_skills)
        
        return SkillAnalysisResponse(
            profile_skills=analysis["profile_skills"],
            missing_skills=analysis["missing_skills"],
            missing_skills_simple=analysis.get("missing_skills_simple"),
            profile_match=analysis["profile_match"],
            recommendations=analysis["recommendations"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing skills: {str(e)}")

# Job recommendations endpoint
@app.post("/api/jobs/recommend")
async def recommend_jobs(request: JobRecommendationRequest) -> JobRecommendationResponse:
    """Recommend jobs based on user skills"""
    try:
        recommendations = skill_analyzer.recommend_jobs(
            request.user_skills, 
            request.limit
        )
        
        return JobRecommendationResponse(
            recommendations=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recommending jobs: {str(e)}")

# Salary trends endpoint
@app.get("/api/trends/salary")
async def get_salary_trends(months_back: int = 24) -> SalaryTrendResponse:
    """Get salary trends over time"""
    try:
        trends = time_analyzer.generate_salary_trends(months_back)
        
        return SalaryTrendResponse(
            dates=trends["dates"],
            values=[float(v) for v in trends["values"]],
            metric=trends["metric"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating salary trends: {str(e)}")

# Technology trends endpoint
@app.get("/api/trends/technology")
async def get_technology_trends(months_back: int = 24) -> TechnologyTrendResponse:
    """Get technology popularity trends"""
    try:
        trends = time_analyzer.generate_technology_trends(months_back)
        
        return TechnologyTrendResponse(
            trends=trends
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating technology trends: {str(e)}")

# CV analysis and enhancement endpoint - updated to match frontend expectations
@app.post("/api/cv/enhance")
async def enhance_cv(request: CVAnalysisRequest) -> EnhancedCVResponse:
    """Analyze and enhance CV"""
    try:
        # Extract skills from CV text
        cv_text = request.cv_text
        common_skills = ['python', 'sql', 'machine learning', 'data science', 'r', 'java', 'javascript', 
                        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
                        'tableau', 'power bi', 'excel', 'aws', 'gcp', 'azure', 'docker', 'kubernetes',
                        'git', 'linux', 'mongodb', 'postgresql', 'mysql', 'hadoop', 'spark']
        
        extracted_skills = []
        cv_lower = cv_text.lower()
        for skill in common_skills:
            if skill in cv_lower:
                extracted_skills.append(skill)
        
        # Analyze CV
        cv_analysis = {"text": cv_text}  # Simplified analysis
        
        # Calculate ATS score
        ats_keywords = ['experience', 'skills', 'education', 'projects', 'achievements', 'responsibilities']
        ats_score = sum(1 for keyword in ats_keywords if keyword in cv_lower) / len(ats_keywords) * 100
        
        # Enhance CV (simplified)
        enhanced_cv = f"Enhanced CV based on: {cv_text[:100]}..."
        
        # Generate LaTeX version (simplified)
        latex_cv = f"\\documentclass{{article}}\n\\begin{{document}}\n{cv_text[:100]}...\n\\end{{document}}"
        
        # Include market trends if requested
        market_trends = None
        if request.include_market_analysis:
            try:
                market_trends = {"trend": "stable", "insight": "Market insights unavailable"}
            except:
                market_trends = {"error": "Market insights unavailable"}
        
        # Include skill recommendations if requested
        skill_recommendations = None
        if request.include_market_analysis:
            try:
                skill_recommendations = [{"skill": "Python", "importance": 0.9}]
            except:
                skill_recommendations = [{"error": "Skill recommendations unavailable"}]
        
        return EnhancedCVResponse(
            extracted_text=", ".join(extracted_skills),
            cv_analysis=cv_analysis,
            ats_score=ats_score,
            enhanced_cv=enhanced_cv,
            latex_cv=latex_cv,
            market_trends=market_trends,
            skill_recommendations=skill_recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing CV: {str(e)}")

# CV Image OCR Processing endpoint
@app.post("/api/cv/process-enhanced")
async def process_cv_image_enhanced(request: CVImageRequest) -> EnhancedCVResponse:
    """Process CV image with OCR and generate comprehensive analysis"""
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image_data)  # Changed from image_base64 to image_data
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        # Process image with OCR
        extracted_text = ocr_processor.process_image(image_data)
        
        # Analyze CV content
        cv_analysis = ocr_processor.analyze_cv_content(extracted_text)
        
        # Calculate ATS score
        ats_result = ocr_processor.calculate_ats_score(cv_analysis)
        ats_score = ats_result.get('percentage', 0)
        
        # Generate enhanced CV
        enhanced_cv = ocr_processor.generate_enhanced_cv(cv_analysis, request.target_role)
        
        # Generate LaTeX CV
        latex_cv = ocr_processor.generate_latex_cv(cv_analysis, request.target_role)
        
        # Generate market trends
        market_trends = ocr_processor.generate_market_trends(request.target_role)
        
        # Generate skill recommendations
        skill_recommendations = ocr_processor.generate_skill_recommendations(cv_analysis, request.target_role)
        
        return EnhancedCVResponse(
            extracted_text=extracted_text,
            cv_analysis={
                "sections": cv_analysis.get('sections_identified', []),
                "skills": cv_analysis.get('extracted_skills', []),
                "experience_years": cv_analysis.get('experience_years', 0),
                "contact_info": cv_analysis.get('contact_info', {}),
                "ats_breakdown": ats_result.get('breakdown', {}),
                "ats_recommendations": ats_result.get('recommendations', [])
            },
            ats_score=ats_result,  # Return full ATS result instead of just percentage
            enhanced_cv=enhanced_cv,
            latex_cv=latex_cv,
            market_trends=market_trends,
            skill_recommendations=skill_recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing CV image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CV image: {str(e)}")

# Comprehensive predictions endpoint
@app.get("/api/predictions")
async def get_comprehensive_predictions(months_back: int = 24, months_ahead: int = 12) -> PredictionResponse:
    """Get comprehensive predictions including trends and forecasts"""
    try:
        # Get current salary trends using the time analyzer
        salary_trend_data = time_analyzer.generate_salary_trends(months_back)
        
        # Get current technology trends
        tech_trend_data = time_analyzer.generate_technology_trends(months_back)
        
        # Generate salary forecast (future predictions)
        salary_forecast_data = time_analyzer.generate_salary_forecast(months_ahead)
        
        # Generate technology forecast
        tech_forecast_data = time_analyzer.generate_technology_forecast(months_ahead)
        
        # Combine all predictions
        predictions = {
            "salary_trends": salary_trend_data,
            "technology_trends": tech_trend_data,
            "salary_forecast": salary_forecast_data,
            "technology_forecast": tech_forecast_data
        }
        
        return PredictionResponse(
            predictions=predictions
        )
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

# Graph-RAG query endpoint
@app.post("/api/rag/query")
async def query_rag(request: RAGQueryRequest) -> RAGResponse:
    """Query the Graph-RAG system"""
    try:
        # Extract question from request body
        question = request.question
        
        if not question:
            raise HTTPException(status_code=400, detail="Question field is required")
        
        # Use the GraphRAG system to answer the question
        result = rag_system.answer_question(question)
        
        # Extract answer and sources
        answer = result.get("answer", "No answer generated")
        sources = [str(source.get("node", "")) for source in result.get("sources", [])]
        
        # Calculate confidence based on number of sources found
        confidence = min(len(sources) / 10.0, 1.0)  # Max confidence of 1.0
        
        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Advanced Graph-RAG query endpoint
@app.post("/api/rag/advanced")
async def advanced_query_rag(request: RAGQueryRequest) -> RAGResponse:
    """Advanced Graph-RAG question answering with better context"""
    try:
        # Extract question from request body
        question = request.question
        
        if not question:
            raise HTTPException(status_code=400, detail="Question field is required")
        
        # Use the GraphRAG system to answer the question with more context
        result = rag_system.answer_question(question, k=20)  # Get more context
        
        # Extract answer and sources
        answer = result.get("answer", "No answer generated")
        sources = [str(source.get("node", "")) for source in result.get("sources", [])]
        
        # Calculate confidence based on number of sources found and quality of answer
        base_confidence = min(len(sources) / 20.0, 1.0)  # Max confidence of 1.0
        # Adjust confidence based on answer quality
        if "could not find enough information" in answer.lower():
            confidence = base_confidence * 0.5  # Lower confidence if no info found
        else:
            confidence = base_confidence * 0.9  # High confidence if info found
        
        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing advanced query: {str(e)}")

# Job prediction endpoint
@app.post("/api/predictions/job")
async def predict_job_outcomes(request: JobPredictionRequest) -> JobPredictionResponse:
    """Predict job market outcomes for specific roles"""
    try:
        job_title = request.job_title
        prediction_type = request.prediction_type
        
        # Generate prediction based on type
        prediction_data = {}
        
        if prediction_type == "salary":
            # Predict salary range
            prediction_data = {
                "predicted_min": 80000,
                "predicted_max": 150000,
                "currency": "USD",
                "growth_rate": 0.075  # 7.5% as a decimal
            }
        elif prediction_type == "growth":
            # Predict growth prospects
            prediction_data = {
                "growth_trend": "increasing",
                "projected_growth_5y": 0.175,  # 17.5% as a decimal
                "market_demand": "high"
            }
        elif prediction_type == "demand":
            # Predict demand level
            prediction_data = {
                "current_demand": "high",
                "seasonal_variations": ["Q1", "Q4"],
                "related_roles": ["Senior Data Scientist", "ML Engineer", "AI Specialist"]
            }
        elif prediction_type == "skills":
            # Predict required skills
            prediction_data = {
                "core_skills": ["Python", "SQL", "Machine Learning"],
                "emerging_skills": ["MLOps", "LLM", "Cloud Platforms"],
                "certifications": ["AWS Certified ML", "Google Professional Data Engineer"]
            }
        else:
            # Default prediction
            prediction_data = {
                "general_outlook": "positive",
                "key_factors": ["AI adoption", "data-driven decision making", "digital transformation"],
                "recommendations": ["Upskill in cloud platforms", "Learn MLOps", "Gain domain expertise"],
                "growth_rate": 0.12  # 12% as a decimal
            }
        
        # Add pre-employment detection for all prediction types (if not already added)
        if "is_pre_employment" not in prediction_data:
            is_pre_employment = any(keyword in job_title.lower() for keyword in [
                "junior", "entry", "intern", "graduate", "trainee", "assistant", "associate"
            ])
            
            # Store as boolean string
            prediction_data["is_pre_employment"] = str(is_pre_employment).lower()
            if is_pre_employment:
                # Store indicators as list
                indicators = [str(keyword) for keyword in [
                    "junior", "entry", "intern", "graduate", "trainee", "assistant", "associate"
                ] if keyword in job_title.lower()]
                prediction_data["pre_employment_indicators"] = indicators
        
        return JobPredictionResponse(
            job_title=job_title,
            prediction_type=prediction_type,
            prediction=prediction_data,
            confidence=0.85  # Generic confidence score
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating job prediction: {str(e)}")

# Skill growth projections endpoint
@app.get("/api/predictions/skill-growth")
async def get_skill_growth_projections(months_ahead: int = 12):
    """Get individual skill growth projections for visualization"""
    try:
        tech_forecasts = time_analyzer.generate_technology_forecast(months_ahead)
        
        # Format each technology as a separate chart-ready object
        skill_projections = []
        for tech in tech_forecasts:
            skill_projections.append({
                "skill_name": tech["metric"].replace(" Demand Forecast", ""),
                "dates": tech["dates"],
                "values": tech["values"],
                "growth_rate": tech["growth_rate"],
                "current_demand": tech["values"][0] if tech["values"] else 0,
                "projected_demand": tech["values"][-1] if tech["values"] else 0
            })
        
        return {
            "skill_projections": skill_projections,
            "total_skills": len(skill_projections),
            "forecast_period_months": months_ahead
        }
    except Exception as e:
        logger.error(f"Error generating skill growth projections: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating skill projections: {str(e)}")

# Technology trends endpoint (historical)
@app.get("/api/predictions/technology-trends")
async def get_technology_trends(months_back: int = 24):
    """Get historical technology trends for individual visualization"""
    try:
        tech_trends = time_analyzer.generate_technology_trends(months_back)
        
        # Format each technology as a separate chart-ready object
        technology_trends = []
        for tech in tech_trends:
            technology_trends.append({
                "technology_name": tech["metric"].replace(" Popularity", ""),
                "dates": tech["dates"],
                "values": tech["values"],
                "current_popularity": tech["values"][-1] if tech["values"] else 0,
                "starting_popularity": tech["values"][0] if tech["values"] else 0,
                "trend": "increasing" if tech["values"][-1] > tech["values"][0] else "decreasing"
            })
        
        return {
            "technology_trends": technology_trends,
            "total_technologies": len(technology_trends),
            "period_months": months_back
        }
    except Exception as e:
        logger.error(f"Error generating technology trends: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating technology trends: {str(e)}")

# Only run the server if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)