"""
Smart CV Page for Job Analysis Dashboard

This page provides OCR-based CV processing with AI-powered recommendations.
"""

import streamlit as st
import pandas as pd
import json
import os
import base64
import requests
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Smart CV",
    page_icon="🧠",
    layout="wide"
)

def main():
    """Main function for the Smart CV page"""
    st.title("🧠 Smart CV Enhancement")
    
    # Description
    st.markdown("""
    Enhance your CV with AI-powered recommendations. Upload your CV image to get:
    - OCR-based text extraction
    - ATS (Applicant Tracking System) scoring
    - Personalized improvement suggestions
    - LaTeX formatted CV generation
    - Market trend analysis
    - Skill development recommendations
    """)
    
    # Warning about Tesseract requirement
    st.warning("""
    **Important**: This feature requires Tesseract OCR to be installed on your system.
    If you haven't installed it yet, please download and install Tesseract OCR from 
    [GitHub](https://github.com/UB-Mannheim/tesseract/wiki) and add it to your system PATH.
    """)
    
    # Create tab for image upload only (removed text input)
    tab1 = st.tabs(["Image Upload"])[0]
    
    with tab1:
        st.markdown("#### Upload CV Image:")
        uploaded_file = st.file_uploader("Upload CV Image", type=["png", "jpg", "jpeg"])
        
        # Target role selection
        target_role = st.selectbox(
            "Target Role (Optional):",
            ["Data Scientist", "Data Analyst", "Machine Learning Engineer", 
             "Business Analyst", "Software Engineer", "Product Manager"],
            index=0,
            key="image_target_role"
        )
        
        # Enhanced analysis options
        st.markdown("#### Analysis Options:")
        col1, col2 = st.columns(2)
        with col1:
            include_trends = st.checkbox("Include Market Trends", value=True)
        with col2:
            include_recommendations = st.checkbox("Include Skill Recommendations", value=True)
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded CV Image", use_column_width=True)
            
            if st.button("Process CV Image"):
                with st.spinner("Processing your CV image..."):
                    try:
                        # Read the image file
                        image_bytes = uploaded_file.read()
                        
                        # Encode image to base64
                        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                        
                        # Prepare the request to our enhanced API endpoint
                        api_url = "http://localhost:8005/api/cv/process-enhanced"
                        payload = {
                            "image_data": encoded_image,
                            "target_role": target_role,
                            "include_trends": include_trends,
                            "include_market_analysis": include_recommendations
                        }
                        
                        # Make API request
                        response = requests.post(api_url, json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Display results
                            st.success("CV successfully processed!")
                            
                            # Create tabs for different sections
                            result_tabs = ["OCR Results", "Enhanced CV", "ATS Score Analysis", "CV Analysis"]
                            if include_trends:
                                result_tabs.append("Market Trends")
                            if include_recommendations:
                                result_tabs.append("Skill Recommendations")
                                
                            tabs = st.tabs(result_tabs)
                            
                            # OCR Results Tab
                            with tabs[0]:
                                st.markdown("#### Extracted Text")
                                st.text_area("Extracted text from CV image:", 
                                           value=result.get("extracted_text", ""),
                                           height=300,
                                           key="extracted_text")
                                
                                # Copy button for extracted text
                                if st.button("Copy Extracted Text", key="copy_extracted"):
                                    st.session_state.clipboard = result.get("extracted_text", "")
                                    st.success("Text copied to clipboard!")
                            
                            # Enhanced CV Tab - Improved styling without text area
                            with tabs[1]:
                                st.markdown("#### Action-Oriented CV Improvement Plan")
                                enhanced_cv = result.get("enhanced_cv", "")
                                
                                # Display enhanced CV with better formatting
                                st.markdown(enhanced_cv)
                                
                                # Copy button for enhanced CV
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("Copy Enhancement Plan", key="copy_enhanced"):
                                        st.session_state.clipboard = enhanced_cv
                                        st.success("Enhancement plan copied to clipboard!")
                                with col2:
                                    # Download as text file
                                    st.download_button(
                                        label="Download as Text",
                                        data=enhanced_cv,
                                        file_name="cv_enhancement_plan.txt",
                                        mime="text/plain"
                                    )
                                with col3:
                                    # Download as LaTeX file
                                    latex_cv = result.get("latex_cv", "")
                                    st.download_button(
                                        label="Download LaTeX",
                                        data=latex_cv,
                                        file_name="cv.tex",
                                        mime="application/x-tex"
                                    )
                            
                            # ATS Score Analysis Tab
                            with tabs[2]:
                                st.markdown("#### Overall ATS Score")
                                ats_result = result.get("ats_score", {})
                                ats_percentage = ats_result.get("percentage", 0)
                                ats_score_value = ats_result.get("ats_score", 0)
                                max_score = ats_result.get("max_score", 100)
                                
                                # Display score as a progress bar
                                st.progress(ats_percentage / 100)
                                st.markdown(f"**{ats_percentage}%** - Score: {ats_score_value} / {max_score}")
                                
                                # Display breakdown
                                breakdown = ats_result.get("breakdown", {})
                                for category, details in breakdown.items():
                                    st.markdown(f"#### {category.title()}")
                                    category_score = details.get('score', 0)
                                    category_max = details.get('max', 1)
                                    if category_max > 0:  # Avoid division by zero
                                        st.progress(min(category_score / category_max, 1.0))
                                    st.markdown(f"Score: **{category_score}** / {details.get('max', 0)}")
                                    st.markdown(f"*{details.get('details', '')}*")
                                    st.markdown("---")
                                
                                # Display recommendations
                                st.markdown("#### Recommendations")
                                recommendations = ats_result.get("recommendations", [])
                                if recommendations:
                                    for rec in recommendations:
                                        st.markdown(f"• {rec}")
                                else:
                                    st.info("Your CV is well-optimized! No major improvements needed.")
                            
                            # CV Analysis Tab
                            with tabs[3]:
                                st.markdown("#### CV Analysis")
                                cv_analysis = result.get("cv_analysis", {})
                                
                                # Identified sections
                                st.markdown("##### Identified Sections")
                                sections = cv_analysis.get("sections_identified", [])
                                section_cols = st.columns(min(len(sections), 5))
                                for i, section in enumerate(sections):
                                    if i < len(section_cols):
                                        with section_cols[i]:
                                            st.markdown(f"**{section}**")
                                
                                # Extracted skills
                                st.markdown("##### Extracted Skills")
                                skills = cv_analysis.get("extracted_skills", [])
                                skill_cols = st.columns(3)
                                for i, skill in enumerate(skills):
                                    with skill_cols[i % 3]:
                                        st.markdown(f"- {skill}")
                                
                                # Contact information
                                st.markdown("##### Contact Information")
                                contact_info = cv_analysis.get("contact_info", {})
                                if contact_info:
                                    for key, value in contact_info.items():
                                        st.markdown(f"**{key.title()}:** {value}")
                                else:
                                    st.markdown("No contact information found")
                                
                                # Experience information
                                st.markdown("##### Experience")
                                exp_years = cv_analysis.get("experience_years", 0)
                                st.markdown(f"**{exp_years} years** of experience")
                                
                                # CV length
                                st.markdown("##### CV Length")
                                cv_length = cv_analysis.get("cv_length", 0)
                                st.markdown(f"**{cv_length} words**")
                                
                                # Experience indicators
                                st.markdown("##### Experience Indicators")
                                exp_indicators = cv_analysis.get("experience_indicators", [])
                                if exp_indicators:
                                    exp_cols = st.columns(3)
                                    for i, indicator in enumerate(exp_indicators):
                                        with exp_cols[i % 3]:
                                            st.markdown(f"- {indicator}")
                                else:
                                    st.markdown("No experience indicators found")
                            
                            # Market Trends Tab (if included)
                            if include_trends and len(tabs) > 4:
                                with tabs[4]:
                                    st.markdown("#### Market Trends Analysis")
                                    market_trends = result.get("market_trends", {})
                                    
                                    if "error" in market_trends:
                                        st.warning(market_trends["error"])
                                    elif market_trends:
                                        st.markdown("##### High-Demand Skills")
                                        high_demand_skills = market_trends.get("high_demand_skills", [])
                                        cols = st.columns(min(len(high_demand_skills), 3))
                                        for i, skill in enumerate(high_demand_skills):
                                            with cols[i % 3]:
                                                st.markdown(f"**{skill}**")
                                        
                                        st.markdown("##### Emerging Trends")
                                        emerging_trends = market_trends.get("emerging_trends", [])
                                        for trend in emerging_trends:
                                            st.markdown(f"• {trend}")
                                        
                                        st.markdown("##### Salary Information")
                                        st.markdown(f"**Salary Range:** {market_trends.get('salary_range', 'N/A')}")
                                        st.markdown(f"**Growth Rate:** {market_trends.get('growth_rate', 'N/A')}")
                                    else:
                                        st.info("No market trends data available for this role.")
                            
                            # Skill Recommendations Tab (if included)
                            if include_recommendations and len(tabs) > (5 if include_trends else 4):
                                with tabs[-1]:
                                    st.markdown("#### Personalized Skill Recommendations")
                                    skill_recommendations = result.get("skill_recommendations", [])
                                    
                                    if "error" in skill_recommendations:
                                        st.warning(skill_recommendations[0]["error"])
                                    elif skill_recommendations:
                                        for i, rec in enumerate(skill_recommendations):
                                            with st.expander(f"{rec['skill']} - {rec['importance']} Priority"):
                                                st.markdown(f"**Market Demand:** {rec['market_demand']}")
                                                st.markdown(f"**Learning Path:** {rec['learning_path']}")
                                                st.markdown(f"**Time to Learn:** {rec['time_to_learn']}")
                                                st.markdown("**Recommended Resources:**")
                                                for resource in rec.get('resources', [])[:3]:
                                                    st.markdown(f"- [{resource}]({resource})")
                                    else:
                                        st.info("No skill recommendations available.")
                        else:
                            error_message = response.json().get("detail", "Unknown error")
                            st.error(f"Error processing CV: {response.status_code} - {error_message}")
                            
                    except Exception as e:
                        st.error(f"Error processing CV image: {str(e)}")
                        st.info("Make sure Tesseract OCR is installed and added to your system PATH.")
        else:
            st.info("Please upload a CV image (PNG, JPG, JPEG) to get started.")
    
    # Features
    st.markdown("#### Features:")
    st.markdown("""
    - **OCR Processing**: Extract text from CV images using advanced OCR
    - **ATS Scoring**: Get a detailed Applicant Tracking System score
    - **AI-Powered Enhancement**: Receive personalized improvement suggestions
    - **LaTeX Generation**: Download professionally formatted CV in LaTeX
    - **Skill Analysis**: Automatic extraction of technical skills
    - **Experience Detection**: Identify experience indicators and years
    - **Market Trends**: Current industry trends and salary information
    - **Skill Recommendations**: Personalized learning paths and resources
    """)
    
    # How it works
    st.markdown("#### How It Works:")
    st.markdown("""
    1. **Upload**: Submit your CV image (PNG, JPG, JPEG)
    2. **Process**: OCR extracts text from the image
    3. **Analyze**: AI analyzes your CV content and structure
    4. **Score**: Get ATS compatibility score with detailed breakdown
    5. **Enhance**: Receive personalized improvement suggestions
    6. **Trends**: Access current market trends for your target role
    7. **Recommendations**: Get personalized skill development paths
    8. **Download**: Get enhanced CV in text or LaTeX format
    """)

if __name__ == "__main__":
    main()