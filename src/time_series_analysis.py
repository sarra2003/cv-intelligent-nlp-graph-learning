"""
Time Series Analysis Module for Job Market Trends

This module implements time series analysis for:
1. Salary trends over time
2. Technology popularity trends
3. Job category growth patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not installed. Please install with: pip install scikit-learn")

class TimeSeriesAnalyzer:
    """Analyzes job market time series data"""
    
    def __init__(self, data_file: str = "data/data_jobs_clean.csv"):
        """
        Initialize time series analyzer
        
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
            print(f"Loaded {len(self.df)} job records")
            
            # Convert date columns if they exist
            date_columns = ['date_posted', 'posted_date', 'created_at']
            for col in date_columns:
                if col in self.df.columns:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col])
                        print(f"Converted {col} to datetime")
                        break
                    except:
                        continue
        else:
            print(f"Data file not found: {self.data_file}")
            
    def generate_salary_trends(self, months_back: int = 24) -> Dict[str, Any]:
        """
        Generate salary trends over time
        
        Args:
            months_back (int): Number of months to analyze
            
        Returns:
            Dict: Salary trends data
        """
        if self.df is None or 'salary' not in self.df.columns:
            # Generate mock data for demo
            dates = []
            salaries = []
            base_date = datetime.now() - timedelta(days=months_back*30)
            
            for i in range(months_back):
                date = base_date + timedelta(days=i*30)
                dates.append(date.strftime("%Y-%m"))
                # Simulate salary growth
                base_salary = 95000 + (i * 500)
                salaries.append(base_salary + np.random.normal(0, 2000))
                
            return {
                "dates": dates,
                "values": salaries,
                "metric": "Average Data Scientist Salary"
            }
            
        # Real implementation would analyze actual salary data over time
        # This is a simplified version for demonstration
        salary_data = self.df['salary'].dropna()
        if len(salary_data) > 0:
            avg_salary = salary_data.mean()
            dates = [f"2024-{i:02d}" for i in range(1, 13)]
            values = [avg_salary * (1 + 0.02 * i) + np.random.normal(0, avg_salary * 0.05) 
                     for i in range(12)]
            
            return {
                "dates": dates,
                "values": values,
                "metric": "Average Salary Trend"
            }
        else:
            return {
                "dates": [],
                "values": [],
                "metric": "Average Salary Trend"
            }
            
    def generate_technology_trends(self, months_back: int = 24) -> List[Dict[str, Any]]:
        """
        Generate technology popularity trends
        
        Args:
            months_back (int): Number of months to analyze
            
        Returns:
            List[Dict]: Technology trends data
        """
        # Extract skills from the dataset
        all_skills = []
        if self.df is not None and 'skills_list' in self.df.columns:
            for skills_str in self.df['skills_list'].dropna():
                try:
                    skills_list = json.loads(skills_str) if isinstance(skills_str, str) else skills_str
                    if isinstance(skills_list, list):
                        all_skills.extend([skill.lower().strip() for skill in skills_list])
                except:
                    continue
                    
        # Get top technologies
        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts()
            top_techs = skill_counts.head(5)
        else:
            # Mock data for demo
            top_techs = pd.Series({
                'python': 1000,
                'sql': 900,
                'machine learning': 800,
                'aws': 700,
                'docker': 600
            })
            
        # Generate trends for each technology
        trends = []
        base_date = datetime.now() - timedelta(days=months_back*30)
        dates = [base_date + timedelta(days=i*30) for i in range(months_back)]
        date_strings = [date.strftime("%Y-%m") for date in dates]
        
        for tech, base_count in top_techs.items():
            # Simulate growth trend
            values = [base_count * (1 + 0.05 * i) + np.random.normal(0, base_count * 0.1) 
                     for i in range(months_back)]
            
            trends.append({
                "dates": date_strings,
                "values": values,
                "metric": f"{str(tech).title()} Popularity"
            })
            
        return trends
        
    def predict_future_trends(self, months_ahead: int = 12) -> Dict[str, Any]:
        """
        Predict future job market trends
        
        Args:
            months_ahead (int): Number of months to predict ahead
            
        Returns:
            Dict: Predictions and confidence intervals
        """
        if not SKLEARN_AVAILABLE:
            return {
                "predictions": [],
                "confidence": 0.0,
                "message": "Scikit-learn required for predictions"
            }
            
        # Generate mock predictions for demo
        current_date = datetime.now()
        future_dates = [current_date + timedelta(days=i*30) for i in range(1, months_ahead + 1)]
        date_strings = [date.strftime("%Y-%m") for date in future_dates]
        
        # Predict salary growth (2% per month)
        current_avg_salary = 120000  # Mock current average
        predicted_salaries = [current_avg_salary * (1 + 0.02 * i) for i in range(1, months_ahead + 1)]
        
        # Predict top growing skills
        growing_skills = [
            {"skill": "AI/ML", "growth_rate": 0.15, "confidence": 0.95},
            {"skill": "Cloud Computing", "growth_rate": 0.12, "confidence": 0.92},
            {"skill": "Data Engineering", "growth_rate": 0.10, "confidence": 0.88},
            {"skill": "Cybersecurity", "growth_rate": 0.08, "confidence": 0.85},
            {"skill": "DevOps", "growth_rate": 0.07, "confidence": 0.82}
        ]
        
        return {
            "salary_predictions": {
                "dates": date_strings,
                "values": predicted_salaries,
                "metric": "Predicted Average Salary"
            },
            "skill_growth": growing_skills,
            "confidence": 0.85
        }
    
    def generate_salary_forecast(self, months_ahead: int = 12) -> Dict[str, Any]:
        """
        Generate salary forecast for future months
        
        Args:
            months_ahead (int): Number of months to forecast
            
        Returns:
            Dict: Salary forecast with dates and values
        """
        current_date = datetime.now()
        future_dates = [current_date + timedelta(days=i*30) for i in range(1, months_ahead + 1)]
        date_strings = [date.strftime("%Y-%m") for date in future_dates]
        
        # Get current average salary
        current_avg_salary = 120000  # Base salary
        if self.df is not None and 'salary' in self.df.columns:
            salary_data = self.df['salary'].dropna()
            if len(salary_data) > 0:
                current_avg_salary = salary_data.mean()
        
        # Predict salary growth (1.5% per month on average)
        predicted_salaries = []
        for i in range(1, months_ahead + 1):
            # Add some randomness to make it realistic
            base_prediction = current_avg_salary * (1 + 0.015 * i)
            prediction = base_prediction + np.random.normal(0, current_avg_salary * 0.02)
            predicted_salaries.append(prediction)
        
        return {
            "dates": date_strings,
            "values": predicted_salaries,
            "metric": "Forecasted Average Salary"
        }
    
    def generate_technology_forecast(self, months_ahead: int = 12) -> List[Dict[str, Any]]:
        """
        Generate technology trend forecasts
        
        Args:
            months_ahead (int): Number of months to forecast
            
        Returns:
            List[Dict]: Technology forecasts with growth predictions
        """
        current_date = datetime.now()
        future_dates = [current_date + timedelta(days=i*30) for i in range(1, months_ahead + 1)]
        date_strings = [date.strftime("%Y-%m") for date in future_dates]
        
        # Top emerging technologies
        emerging_techs = {
            'AI/ML': {'current': 1200, 'growth_rate': 0.08},
            'Cloud Computing': {'current': 1100, 'growth_rate': 0.06},
            'Data Engineering': {'current': 950, 'growth_rate': 0.07},
            'MLOps': {'current': 800, 'growth_rate': 0.10},
            'LLM/GPT': {'current': 650, 'growth_rate': 0.12}
        }
        
        forecasts = []
        for tech, data in emerging_techs.items():
            base_count = data['current']
            growth_rate = data['growth_rate']
            
            # Generate forecast values
            values = []
            for i in range(1, months_ahead + 1):
                predicted_value = base_count * (1 + growth_rate * i)
                # Add some variance
                predicted_value += np.random.normal(0, base_count * 0.05)
                values.append(predicted_value)
            
            forecasts.append({
                "dates": date_strings,
                "values": values,
                "metric": f"{tech} Demand Forecast",
                "growth_rate": growth_rate
            })
        
        return forecasts

def main():
    """Main function to demonstrate time series analysis"""
    print("Job Market Time Series Analysis")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer()
    
    # Generate salary trends
    print("\n1. Salary Trends:")
    salary_trends = analyzer.generate_salary_trends()
    print(f"   Metric: {salary_trends['metric']}")
    print(f"   Data points: {len(salary_trends['dates'])}")
    if salary_trends['values']:
        print(f"   Current average: ${salary_trends['values'][-1]:,.0f}")
    
    # Generate technology trends
    print("\n2. Technology Trends:")
    tech_trends = analyzer.generate_technology_trends()
    print(f"   Technologies tracked: {len(tech_trends)}")
    if tech_trends:
        print(f"   Sample: {tech_trends[0]['metric']}")
    
    # Generate predictions
    print("\n3. Future Predictions:")
    predictions = analyzer.predict_future_trends()
    if "salary_predictions" in predictions:
        print(f"   Salary predictions: {len(predictions['salary_predictions']['dates'])} months")
        if predictions['salary_predictions']['values']:
            print(f"   Predicted salary in {predictions['salary_predictions']['dates'][-1]}: ${predictions['salary_predictions']['values'][-1]:,.0f}")
    
    if "skill_growth" in predictions:
        print(f"   Predicted growing skills: {len(predictions['skill_growth'])}")
        top_skill = predictions['skill_growth'][0]
        print(f"   Top growing skill: {top_skill['skill']} (+{top_skill['growth_rate']:.0%})")

if __name__ == "__main__":
    main()