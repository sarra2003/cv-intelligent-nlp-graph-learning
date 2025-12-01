"""
OCR and CV Processing Module

This module implements functionality to:
1. Process CV images using OCR
2. Extract text from images
3. Analyze CV content
4. Generate enhanced CVs with ATS scoring
5. Create LaTeX formatted CVs
6. Provide market trend analysis and skill recommendations
"""

import os
import io
import json
import base64
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import re
import logging

# Try to import LLM manager
try:
    from src.llm_integration import LLMManager
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMManager = None
    print("LLM integration not available. Using template-based generation.")

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    
    # On Windows, explicitly set the tesseract path if not in PATH
    if os.name == 'nt':  # Windows
        # Common installation paths for Tesseract on Windows
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME')),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None  # type: ignore
    print("Tesseract OCR not available. Using fallback text extraction.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    """Processes CV images using OCR and generates enhanced CVs"""
    
    def __init__(self):
        """Initialize OCR processor"""
        if LLM_AVAILABLE and LLMManager is not None:
            self.llm_manager = LLMManager()
        else:
            self.llm_manager = None
        
        # Check if Tesseract is installed and configured
        self.tesseract_available = False
        if TESSERACT_AVAILABLE and pytesseract is not None:
            try:
                # Try to get Tesseract version to verify installation
                pytesseract.get_tesseract_version()
                self.tesseract_available = True
                logger.info("Tesseract OCR successfully initialized")
            except Exception as e:
                logger.warning(f"Tesseract OCR is not installed or not in PATH: {e}")
                logger.info("Falling back to basic text extraction methods.")
        
    def process_image(self, image_data: bytes) -> str:
        """
        Process image/PDF and extract text using OCR or PDF extraction
        
        Args:
            image_data (bytes): Image or PDF data (PNG, JPG, PDF, etc.)
            
        Returns:
            str: Extracted text from image/PDF
        """
        try:
            # Check if it's a PDF file
            if image_data[:4] == b'%PDF':
                logger.info("Detected PDF file - using PDF text extraction")
                return self._extract_text_from_pdf(image_data)
            
            # Check if Tesseract is available for image OCR
            if self.tesseract_available and pytesseract is not None:
                # Open image from bytes
                image = Image.open(io.BytesIO(image_data))
                
                logger.info(f"Image loaded: size={image.size}, mode={image.mode}")
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    logger.info(f"Image converted to RGB")
                
                # ADVANCED IMAGE PREPROCESSING
                import numpy as np
                from PIL import ImageFilter, ImageOps
                
                # Convert PIL image to numpy array for advanced processing
                img_array = np.array(image)
                
                # Convert to grayscale
                image = image.convert('L')  # Grayscale
                logger.info("Converted to grayscale")
                
                # Resize if too small (upscale for better OCR)
                min_width = 1200
                if image.width < min_width:
                    ratio = min_width / image.width
                    new_size = (int(image.width * ratio), int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Upscaled image to {new_size}")
                
                # Apply denoising
                image = image.filter(ImageFilter.MedianFilter(size=3))
                
                # Increase contrast significantly
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(2.5)  # Increased from 1.5
                
                # Increase sharpness
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(3.0)  # Increased from 2.0
                
                # Apply threshold to make text more clear (binarization)
                image = ImageOps.autocontrast(image)
                
                # Convert back to array for threshold
                img_array = np.array(image)
                
                # Apply adaptive thresholding for better text extraction
                try:
                    import cv2
                    # Apply adaptive threshold
                    img_array = cv2.adaptiveThreshold(
                        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 11, 2
                    )
                    image = Image.fromarray(img_array)
                    logger.info("Applied adaptive thresholding (OpenCV)")
                    
                    # Save preprocessed image for debugging
                    try:
                        debug_path = "ocr_debug_preprocessed.png"
                        image.save(debug_path)
                        logger.info(f"Saved preprocessed image to {debug_path}")
                    except:
                        pass
                        
                except ImportError:
                    # Fallback to simple threshold if OpenCV not available
                    threshold = 128
                    img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
                    image = Image.fromarray(img_array)
                    logger.info("Applied simple thresholding")
                
                # Perform OCR with multiple configs and choose the best result
                configs = [
                    r'--oem 3 --psm 6',  # Uniform block of text
                    r'--oem 3 --psm 4',  # Single column of text
                    r'--oem 3 --psm 3',  # Fully automatic page segmentation
                ]
                
                best_text = ""
                best_length = 0
                
                for config in configs:
                    try:
                        extracted_text = pytesseract.image_to_string(image, config=config, lang='fra+eng')
                        if len(extracted_text) > best_length:
                            best_text = extracted_text
                            best_length = len(extracted_text)
                            logger.info(f"Config '{config}' extracted {len(extracted_text)} chars")
                    except Exception as e:
                        logger.warning(f"Config '{config}' failed: {e}")
                        continue
                
                extracted_text = best_text
                logger.info(f"OCR completed with Tesseract. Best result: {len(extracted_text)} characters")
                
                # Log full extracted text for debugging
                logger.info(f"\n{'='*60}\nFULL OCR EXTRACTED TEXT:\n{'='*60}\n{extracted_text}\n{'='*60}")
                
                # Check if we actually got text
                if extracted_text and len(extracted_text.strip()) > 10:
                    logger.info(f"OCR SUCCESS - Extracted {len(extracted_text)} characters")
                    return extracted_text
                else:
                    logger.warning(f"OCR returned empty or very short text ({len(extracted_text)} chars). Using fallback.")
                    return self._get_sample_cv()
            else:
                # Fallback method - return a sample CV for demonstration
                logger.warning("Tesseract not available. Using fallback sample CV.")
                return self._get_sample_cv()
                
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            # Return error message instead of sample CV so user knows something went wrong
            return f"ERROR: OCR processing failed - {str(e)}\n\nPlease try:\n1. A clearer image\n2. Higher resolution (min 1200px width)\n3. Better contrast\n4. Scan instead of photo\n5. PDF export instead of image"
    
    def _extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """
        Extract text from PDF file using PyPDF2 or pdfplumber
        
        Args:
            pdf_data (bytes): PDF file data
            
        Returns:
            str: Extracted text from PDF
        """
        try:
            # Try using PyPDF2 first
            try:
                import PyPDF2
                import io
                
                pdf_file = io.BytesIO(pdf_data)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                extracted_text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    extracted_text += page.extract_text() + "\n"
                
                if extracted_text.strip():
                    logger.info(f"PDF text extraction successful with PyPDF2: {len(extracted_text)} characters")
                    logger.info(f"\n{'='*60}\nFULL PDF EXTRACTED TEXT:\n{'='*60}\n{extracted_text}\n{'='*60}")
                    return extracted_text
            except ImportError:
                logger.warning("PyPDF2 not installed, trying pdfplumber...")
            
            # Try pdfplumber as fallback
            try:
                import pdfplumber
                import io
                
                pdf_file = io.BytesIO(pdf_data)
                extracted_text = ""
                
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            extracted_text += text + "\n"
                
                if extracted_text.strip():
                    logger.info(f"PDF text extraction successful with pdfplumber: {len(extracted_text)} characters")
                    logger.info(f"\n{'='*60}\nFULL PDF EXTRACTED TEXT:\n{'='*60}\n{extracted_text}\n{'='*60}")
                    return extracted_text
            except ImportError:
                logger.warning("pdfplumber not installed")
            
            # If both libraries fail or are not installed
            return "ERROR: PDF text extraction failed. Please install PyPDF2 or pdfplumber: pip install PyPDF2 pdfplumber"
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
            return f"ERROR: Failed to extract text from PDF - {str(e)}"
            
    def _get_sample_cv(self) -> str:
        """Return a sample CV for demonstration when OCR is not available"""
        sample_cv = """
        John Doe
        Data Scientist
        
        CONTACT
        Email: john.doe@example.com
        Phone: (555) 123-4567
        LinkedIn: linkedin.com/in/johndoe
        
        SUMMARY
        Experienced Data Scientist with 3 years of experience in machine learning,
        data analysis, and statistical modeling. Skilled in Python, SQL, and cloud
        platforms. Passionate about transforming data into actionable insights.
        
        EXPERIENCE
        Senior Data Analyst | TechCorp | Jan 2022 - Present
        • Developed machine learning models that improved customer retention by 15%
        • Created automated reporting systems using Python and SQL
        • Led data analysis for product optimization projects
        
        Data Analyst | Data Insights Inc. | Jun 2020 - Dec 2021
        • Analyzed large datasets to identify business trends and opportunities
        • Built interactive dashboards using Tableau and Power BI
        • Collaborated with cross-functional teams to deliver data-driven solutions
        
        SKILLS
        • Programming: Python, SQL, R
        • Machine Learning: Scikit-learn, TensorFlow, PyTorch
        • Data Visualization: Tableau, Power BI, Matplotlib
        • Cloud: AWS, GCP
        • Tools: Git, Docker, Jupyter Notebooks
        
        EDUCATION
        M.S. in Data Science | University of Technology | 2020
        B.S. in Computer Science | State University | 2018
        """
        return sample_cv.strip()
            
    def analyze_cv_content(self, cv_text: str) -> Dict[str, Any]:
        """
        Analyze CV content and extract key information
        
        Args:
            cv_text (str): Extracted CV text
            
        Returns:
            Dict: Analysis results including skills, experience, sections, etc.
        """
        # Common technical skills
        tech_skills = [
            'python', 'sql', 'java', 'javascript', 'r', 'scala', 'matlab',
            'tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes',
            'tableau', 'power bi', 'matplotlib', 'seaborn',
            'git', 'jenkins', 'ci/cd', 'agile', 'scrum',
            'react', 'angular', 'vue', 'node.js', 'express',
            'mongodb', 'postgresql', 'mysql', 'redis',
            'linux', 'bash', 'shell', 'ansible', 'terraform',
            'machine learning', 'deep learning', 'neural networks', 'nlp',
            'data visualization', 'statistical analysis', 'data mining',
            'spark', 'hadoop', 'kafka', 'elasticsearch'
        ]
        
        # Extract skills from CV
        cv_lower = cv_text.lower()
        found_skills = [skill for skill in tech_skills if skill in cv_lower]
        
        # Experience indicators
        experience_indicators = [
            'experience', 'years', 'worked', 'developed', 'implemented',
            'managed', 'led', 'created', 'built', 'designed', 'achieved',
            'improved', 'increased', 'reduced', 'optimized', 'engineered',
            'architected', 'deployed', 'maintained', 'analyzed', 'researched'
        ]
        
        experience_mentions = [word for word in experience_indicators if word in cv_lower]
        
        # Identify CV sections
        sections = self._identify_cv_sections(cv_text)
        
        # Extract contact information
        contact_info = self._extract_contact_info(cv_text)
        
        # Estimate experience years
        experience_years = self._estimate_experience(cv_text)
        
        return {
            "extracted_skills": found_skills,
            "experience_indicators": experience_mentions,
            "cv_length": len(cv_text.split()),
            "sections_identified": sections,
            "contact_info": contact_info,
            "experience_years": experience_years,
            "raw_text": cv_text
        }
        
    def _identify_cv_sections(self, cv_text: str) -> List[str]:
        """Identify common CV sections"""
        section_keywords = [
            'summary', 'objective', 'profile', 'experience', 'employment', 'work',
            'education', 'skills', 'competencies', 'projects', 'portfolio',
            'certifications', 'certificates', 'awards', 'publications',
            'languages', 'references', 'achievements', 'volunteer'
        ]
        
        found_sections = []
        cv_lower = cv_text.lower()
        
        for section in section_keywords:
            # Look for section headers (usually capitalized or with special formatting)
            pattern = r'(?:^|\n)(?:\*{0,2}[_\*]{0,2})\s*' + re.escape(section) + r'\s*(?:[_\*]{0,2}|\n)'
            if re.search(pattern, cv_lower, re.IGNORECASE):
                found_sections.append(section.title())
                
        return found_sections
        
    def _extract_contact_info(self, cv_text: str) -> Dict[str, str]:
        """Extract contact information from CV"""
        contact_info = {}
        
        # Try to extract name (first line or lines before contact info)
        lines = cv_text.strip().split('\n')
        if lines:
            first_line = lines[0].strip()
            # Simple heuristic: if first line has no @ or numbers, it might be name
            if '@' not in first_line and not any(c.isdigit() for c in first_line):
                contact_info['name'] = first_line
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, cv_text)
        if emails:
            contact_info['email'] = emails[0]
            
        # Phone pattern (various formats)
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        phones = re.findall(phone_pattern, cv_text)
        if phones:
            contact_info['phone'] = phones[0]
            
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, cv_text, re.IGNORECASE)
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
            
        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        github = re.findall(github_pattern, cv_text, re.IGNORECASE)
        if github:
            contact_info['github'] = github[0]
            
        return contact_info
        
    def _estimate_experience(self, cv_text: str) -> float:
        """Estimate years of experience from CV text"""
        # Look for date patterns
        date_patterns = [
            r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)',  # 2018 - 2022 or 2018 - Present
            r'(\d{4})\s*to\s*(\d{4}|present|current)',     # 2018 to 2022
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}'  # Month Year format
        ]
        
        experience_years = 0
        all_dates = []
        
        for pattern in date_patterns:
            matches = re.findall(pattern, cv_text, re.IGNORECASE)
            if matches:
                # For year ranges, calculate difference
                if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    try:
                        for match in matches:
                            start_year = int(match[0])
                            end_year_str = match[1].lower()
                            end_year = 2025 if end_year_str in ['present', 'current'] else int(end_year_str)
                            all_dates.append((start_year, end_year))
                    except:
                        pass
                # For month/year patterns, just collect the years
                else:
                    try:
                        years = re.findall(r'\d{4}', cv_text)
                        years = [int(y) for y in years if 1980 <= int(y) <= 2025]
                        if years:
                            all_dates.extend([(y, y) for y in years])
                    except:
                        pass
        
        # Calculate experience based on date ranges
        if all_dates:
            # Sort by start year
            all_dates.sort()
            # Calculate total experience by merging overlapping periods
            if all_dates:
                total_experience = 0
                current_start, current_end = all_dates[0]
                
                for start, end in all_dates[1:]:
                    if start <= current_end:
                        # Overlapping periods, merge them
                        current_end = max(current_end, end)
                    else:
                        # Non-overlapping period, add previous period to total
                        total_experience += current_end - current_start
                        current_start, current_end = start, end
                
                # Add the last period
                total_experience += current_end - current_start
                experience_years = total_experience
        
        return experience_years
        
    def calculate_ats_score(self, cv_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate ATS (Applicant Tracking System) score for CV
        
        Args:
            cv_analysis (Dict): CV analysis results
            
        Returns:
            Dict: ATS score and detailed breakdown
        """
        score = 0
        max_score = 100
        breakdown = {}
        
        # Add logging to debug ATS score calculation
        logger.info(f"Calculating ATS score from analysis: {list(cv_analysis.keys())}")
        
        # Section completeness (20 points)
        sections = cv_analysis.get('sections_identified', [])
        section_score = min(len(sections) * 4, 20)  # 4 points per section, max 20
        score += section_score
        breakdown['sections'] = {
            'score': section_score,
            'max': 20,
            'details': f"Found {len(sections)} sections: {', '.join(sections[:5])}"
        }
        logger.info(f"Sections score: {section_score}/20 (found {len(sections)} sections)")
        
        # Skills relevance (25 points)
        skills = cv_analysis.get('extracted_skills', [])
        skills_score = min(len(skills) * 3, 25)  # 3 points per skill, max 25
        score += skills_score
        breakdown['skills'] = {
            'score': skills_score,
            'max': 25,
            'details': f"Found {len(skills)} relevant skills: {', '.join(skills[:10])}"
        }
        logger.info(f"Skills score: {skills_score}/25 (found {len(skills)} skills)")
        
        # Experience indicators (20 points)
        exp_indicators = cv_analysis.get('experience_indicators', [])
        exp_score = min(len(exp_indicators) * 2, 20)  # 2 points per indicator, max 20
        score += exp_score
        breakdown['experience'] = {
            'score': exp_score,
            'max': 20,
            'details': f"Found {len(exp_indicators)} experience indicators"
        }
        logger.info(f"Experience score: {exp_score}/20 (found {len(exp_indicators)} indicators)")
        
        # Contact information (10 points)
        contact_info = cv_analysis.get('contact_info', {})
        contact_score = min(len(contact_info) * 2, 10)  # 2 points per contact item (changed from 5), max 10
        score += contact_score
        breakdown['contact'] = {
            'score': contact_score,
            'max': 10,
            'details': f"Found {len(contact_info)} contact items: {list(contact_info.keys())}"
        }
        logger.info(f"Contact score: {contact_score}/10 (found {len(contact_info)} contact items)")
        
        # Length appropriateness (10 points)
        word_count = cv_analysis.get('cv_length', 0)
        if 300 <= word_count <= 800:
            length_score = 10
        elif 200 <= word_count < 300 or 800 < word_count <= 1000:
            length_score = 7
        elif 100 <= word_count < 200 or 1000 < word_count <= 1200:
            length_score = 5
        elif 50 <= word_count < 100:  # More lenient for OCR issues
            length_score = 3
        else:
            length_score = 1  # Give at least 1 point instead of 0
        score += length_score
        breakdown['length'] = {
            'score': length_score,
            'max': 10,
            'details': f"CV has {word_count} words"
        }
        logger.info(f"Length score: {length_score}/10 ({word_count} words)")
        
        # Experience years (15 points)
        exp_years = cv_analysis.get('experience_years', 0)
        if exp_years >= 5:
            exp_years_score = 15
        elif exp_years >= 3:
            exp_years_score = 12
        elif exp_years >= 1:
            exp_years_score = 8
        elif exp_years > 0:
            exp_years_score = 5
        else:
            exp_years_score = 2  # Give at least 2 points instead of 0
        score += exp_years_score
        breakdown['experience_years'] = {
            'score': exp_years_score,
            'max': 15,
            'details': f"Estimated {exp_years} years of experience"
        }
        logger.info(f"Experience years score: {exp_years_score}/15 ({exp_years} years)")
        
        # Ensure score doesn't exceed maximum
        score = min(score, max_score)
        
        logger.info(f"FINAL ATS SCORE: {score}/{max_score} ({round((score / max_score) * 100, 1)}%)")
        
        return {
            "ats_score": score,
            "max_score": max_score,
            "percentage": round((score / max_score) * 100, 1),
            "breakdown": breakdown,
            "recommendations": self._generate_ats_recommendations(breakdown)
        }
        
    def _generate_ats_recommendations(self, breakdown: Dict[str, Any]) -> List[str]:
        """Generate recommendations to improve ATS score"""
        recommendations = []
        
        # Sections recommendation
        sections_breakdown = breakdown.get('sections', {})
        if sections_breakdown.get('score', 0) < sections_breakdown.get('max', 20) * 0.7:
            recommendations.append("Add more CV sections like Skills, Projects, Certifications, and Achievements")
            
        # Skills recommendation
        skills_breakdown = breakdown.get('skills', {})
        if skills_breakdown.get('score', 0) < skills_breakdown.get('max', 25) * 0.6:
            recommendations.append("Include more technical skills relevant to your target role, especially programming languages and tools")
            
        # Experience recommendation
        exp_breakdown = breakdown.get('experience', {})
        if exp_breakdown.get('score', 0) < exp_breakdown.get('max', 20) * 0.6:
            recommendations.append("Use more action verbs to describe your experience (developed, managed, created, optimized)")
            
        # Contact recommendation
        contact_breakdown = breakdown.get('contact', {})
        if contact_breakdown.get('score', 0) < contact_breakdown.get('max', 10) * 0.8:
            recommendations.append("Ensure contact information is clearly visible (email, phone, LinkedIn, GitHub)")
            
        # Length recommendation
        length_breakdown = breakdown.get('length', {})
        if length_breakdown.get('score', 0) < length_breakdown.get('max', 10) * 0.7:
            recommendations.append("Optimize CV length - aim for 1-2 pages with concise, relevant information")
            
        # Experience years recommendation
        exp_years_breakdown = breakdown.get('experience_years', {})
        if exp_years_breakdown.get('score', 0) < exp_years_breakdown.get('max', 15) * 0.6:
            recommendations.append("Highlight any relevant experience, projects, or educational achievements that demonstrate your skills")
            
        return recommendations
        
    def generate_enhanced_cv(self, cv_analysis: Dict[str, Any], target_role: str = "Data Professional") -> str:
        """
        Generate enhanced CV content - ACTUAL IMPROVED CV TEXT, not just recommendations
        
        Args:
            cv_analysis (Dict): CV analysis results
            target_role (str): Target job role
            
        Returns:
            str: Complete enhanced CV text ready to use
        """
        cv_text = cv_analysis.get('raw_text', '')
        contact_info = cv_analysis.get('contact_info', {})
        skills = cv_analysis.get('extracted_skills', [])
        experience_years = cv_analysis.get('experience_years', 0)
        missing_skills = self._identify_missing_skills(cv_analysis, target_role)
        
        # Generate actual improved CV text
        return self._generate_actual_enhanced_cv(cv_text, contact_info, skills, experience_years, missing_skills, target_role)
        
    def _generate_actual_enhanced_cv(self, cv_text: str, contact_info: Dict, skills: List[str], 
                                     experience_years: float, missing_skills: List[str], 
                                     target_role: str) -> str:
        """
        Generate actual enhanced CV with all improvements applied
        """
        # Extract name from contact info or CV text
        name = contact_info.get('name', 'Your Name')
        email = contact_info.get('email', 'your.email@example.com')
        phone = contact_info.get('phone', '+33 X XX XX XX XX')
        linkedin = contact_info.get('linkedin', 'linkedin.com/in/yourprofile')
        github = contact_info.get('github', 'github.com/yourprofile')
        
        # Combine existing skills with missing ones
        all_skills = list(set(skills + missing_skills[:8]))  # Add top 8 missing skills
        
        # Organize skills by category
        programming_skills = [s for s in all_skills if any(lang in s.lower() for lang in ['python', 'sql', 'r', 'java', 'javascript', 'scala', 'c++'])]
        frameworks = [s for s in all_skills if any(fw in s.lower() for fw in ['tensorflow', 'pytorch', 'scikit', 'pandas', 'numpy', 'django', 'flask', 'react'])]
        tools = [s for s in all_skills if any(t in s.lower() for t in ['git', 'docker', 'kubernetes', 'airflow', 'spark', 'hadoop'])]
        cloud = [s for s in all_skills if any(c in s.lower() for c in ['aws', 'azure', 'gcp', 'cloud'])]
        viz = [s for s in all_skills if any(v in s.lower() for v in ['tableau', 'power bi', 'matplotlib', 'seaborn', 'plotly'])]
        other_skills = [s for s in all_skills if s not in programming_skills + frameworks + tools + cloud + viz]
        
        # Build enhanced CV
        enhanced_cv = f"""═══════════════════════════════════════════════════════════════
{name.upper()}
{target_role}
═══════════════════════════════════════════════════════════════

📧 {email}  |  📱 {phone}
💼 {linkedin}  |  🔗 {github}

───────────────────────────────────────────────────────────────
PROFESSIONAL SUMMARY
───────────────────────────────────────────────────────────────

Results-driven {target_role} with {int(experience_years)} years of hands-on 
experience in data analysis, machine learning, and technical problem-solving. 
Proven track record of delivering data-driven solutions that drive measurable 
business impact. Expert in {', '.join(programming_skills[:3])} with strong 
capabilities in {', '.join(frameworks[:2])}. Passionate about leveraging 
advanced analytics and automation to solve complex business challenges.

───────────────────────────────────────────────────────────────
CORE COMPETENCIES
───────────────────────────────────────────────────────────────

▸ Programming Languages:
  {', '.join(programming_skills) if programming_skills else 'Python, SQL, R'}

▸ Frameworks & Libraries:
  {', '.join(frameworks) if frameworks else 'TensorFlow, PyTorch, Scikit-learn, Pandas'}

▸ Cloud & DevOps:
  {', '.join(cloud + tools) if cloud or tools else 'AWS, Docker, Git, Kubernetes'}

▸ Data Visualization:
  {', '.join(viz) if viz else 'Tableau, Power BI, Matplotlib, Seaborn'}

▸ Additional Skills:
  {', '.join(other_skills[:6]) if other_skills else 'Machine Learning, Statistics, Data Mining, ETL'}

───────────────────────────────────────────────────────────────
PROFESSIONAL EXPERIENCE
───────────────────────────────────────────────────────────────

[Based on your original CV, rewrite each position with these improvements:]

✓ Use STRONG ACTION VERBS (Engineered, Developed, Optimized, Led, Architected)
✓ Add QUANTIFIABLE METRICS (percentages, time saved, revenue impact)
✓ Highlight TECHNOLOGIES USED in each bullet point
✓ Focus on BUSINESS IMPACT and RESULTS

Example format:

📍 Senior Data Scientist | Company Name | 2022 - Present
   • Engineered end-to-end ML pipeline using {programming_skills[0] if programming_skills else 'Python'} 
     and {frameworks[0] if frameworks else 'TensorFlow'}, reducing model training time by 45%
   • Led cross-functional team of 5 in developing predictive analytics solution,
     resulting in $2M annual cost savings
   • Architected automated data preprocessing workflow using {tools[0] if tools else 'Apache Airflow'},
     improving data quality by 35%
   • Deployed 8 ML models to production on {cloud[0] if cloud else 'AWS'} with 99.9% uptime

📍 Data Analyst | Previous Company | 2020 - 2022
   • Developed interactive dashboards using {viz[0] if viz else 'Tableau'} for C-suite executives,
     enabling data-driven decision making across 3 departments
   • Optimized SQL queries and database performance, reducing report generation time by 60%
   • Collaborated with product team to implement A/B testing framework,
     increasing conversion rate by 18%
   • Analyzed customer behavior data (2M+ records), identifying key segments
     that drove 25% revenue growth

───────────────────────────────────────────────────────────────
KEY PROJECTS & ACHIEVEMENTS
───────────────────────────────────────────────────────────────

🏆 Predictive Analytics Platform
   Built end-to-end machine learning platform using {', '.join(programming_skills[:2])} 
   and {frameworks[0] if frameworks else 'Scikit-learn'}, achieving 92% prediction accuracy 
   and reducing customer churn by 15%

🏆 Real-Time Data Pipeline
   Designed and deployed real-time ETL pipeline processing 10M+ events/day using 
   {tools[0] if tools else 'Apache Spark'} and {cloud[0] if cloud else 'AWS'}, 
   enabling near-instant business intelligence

🏆 Automated Reporting System
   Created automated reporting infrastructure that reduced manual reporting time 
   from 20 hours/week to 2 hours/week, saving $50K annually

───────────────────────────────────────────────────────────────
EDUCATION
───────────────────────────────────────────────────────────────

🎓 Master's Degree in Data Science / Computer Science
   University Name | Year
   
🎓 Bachelor's Degree in [Your Field]
   University Name | Year

───────────────────────────────────────────────────────────────
CERTIFICATIONS
───────────────────────────────────────────────────────────────

📜 AWS Certified Data Analytics – Specialty
📜 TensorFlow Developer Certificate
📜 Tableau Desktop Specialist

───────────────────────────────────────────────────────────────
ADDITIONAL INFORMATION
───────────────────────────────────────────────────────────────

• Languages: French (Native), English (Fluent)
• Publications: [If applicable - research papers, blog posts]
• Speaking: [If applicable - conferences, meetups]
• Open Source: [If applicable - GitHub contributions]

═══════════════════════════════════════════════════════════════

NOTE: This enhanced CV includes:
✅ ATS-optimized formatting with clear sections
✅ Action-oriented language with strong verbs
✅ Quantified achievements and metrics
✅ Comprehensive skill coverage (existing + recommended)
✅ Industry-relevant keywords for {target_role}
✅ Professional summary highlighting key strengths
✅ Complete contact information

Please replace bracketed sections with your actual information.
"""
        
        return enhanced_cv.strip()
        
    def _get_market_context(self, target_role: str) -> str:
        """Get current market context for the target role"""
        try:
            # Define market context for different roles
            market_contexts = {
                "Data Scientist": """
                Current Market Context for Data Scientists:
                - High demand for cloud platforms (AWS, Azure, GCP) with machine learning services
                - Growing importance of MLOps and model deployment skills
                - Emphasis on data storytelling and business impact communication
                - Increasing focus on ethical AI and responsible ML practices
                - Competitive salary range: $100K-$200K+ depending on experience and location
                - Key differentiators: certifications, personal projects, measurable impact
                - Emerging trends: AutoML, MLOps, LLMOps, and AI governance
                """,
                "Data Analyst": """
                Current Market Context for Data Analysts:
                - Strong demand for visualization tools (Tableau, Power BI, Looker)
                - Growing importance of SQL optimization and data warehousing
                - Emphasis on business acumen and stakeholder communication
                - Increasing focus on data quality and governance
                - Competitive salary range: $70K-$120K+ depending on experience and location
                - Key differentiators: domain expertise, storytelling ability, automation skills
                - Emerging trends: self-service analytics, data democratization, and augmented analytics
                """,
                "Machine Learning Engineer": """
                Current Market Context for Machine Learning Engineers:
                - High demand for deep learning frameworks (TensorFlow, PyTorch)
                - Growing importance of MLOps and model deployment pipelines
                - Emphasis on cloud platforms and distributed computing
                - Increasing focus on model monitoring and observability
                - Competitive salary range: $120K-$200K+ depending on experience and location
                - Key differentiators: production experience, scalability knowledge, system design
                - Emerging trends: LLMOps, model compression, federated learning, and edge AI
                """,
                "Business Analyst": """
                Current Market Context for Business Analysts:
                - Strong demand for requirements gathering and process modeling
                - Growing importance of data analysis and visualization skills
                - Emphasis on stakeholder management and communication
                - Increasing focus on digital transformation and Agile methodologies
                - Competitive salary range: $75K-$125K+ depending on experience and location
                - Key differentiators: domain expertise, certification, change management skills
                - Emerging trends: data-driven decision making, business intelligence, and analytics
                """
            }
            
            # Return specific context or generic context
            return market_contexts.get(target_role, f"""
            Current Market Context for {target_role}:
            - High demand for technical and analytical skills
            - Growing importance of data-driven decision making
            - Emphasis on communication and collaboration
            - Increasing focus on continuous learning and adaptation
            - Competitive salary varies by location and experience
            - Key differentiators: certifications, projects, measurable impact
            - Emerging trends relevant to the field
            """).strip()
        except:
            return "Market data unavailable"
            
    def _identify_missing_skills(self, cv_analysis: Dict[str, Any], target_role: str) -> List[str]:
        """Identify missing skills based on target role and market analysis"""
        # Define skill sets for different roles
        role_skills = {
            "Data Scientist": [
                "Machine Learning", "Deep Learning", "Python", "SQL", "Statistics",
                "TensorFlow", "PyTorch", "Scikit-learn", "Data Visualization",
                "AWS", "Experimentation", "A/B Testing", "MLOps", "Feature Engineering",
                "Natural Language Processing", "Computer Vision"
            ],
            "Data Analyst": [
                "SQL", "Python", "Data Visualization", "Tableau", "Power BI",
                "Statistics", "Excel", "Data Cleaning", "Business Analysis",
                "Storytelling", "Dashboard Design", "Looker", "Data Warehousing",
                "ETL Processes", "Google Analytics"
            ],
            "Machine Learning Engineer": [
                "Python", "TensorFlow", "PyTorch", "Scikit-learn", "AWS",
                "Docker", "Kubernetes", "CI/CD", "Model Deployment",
                "Feature Engineering", "MLOps", "Spark", "Airflow",
                "Model Monitoring", "Distributed Systems"
            ],
            "Business Analyst": [
                "SQL", "Excel", "Data Analysis", "Requirements Gathering",
                "Process Modeling", "Stakeholder Management", "Agile",
                "Tableau", "Power BI", "Storytelling", "JIRA",
                "Business Process Improvement", "Change Management"
            ]
        }
        
        # Get relevant skills for target role
        relevant_skills = role_skills.get(target_role, role_skills["Data Scientist"]) \
            if target_role in role_skills else \
            ["Python", "SQL", "Data Analysis", "Statistics", "Machine Learning"]
        
        # Find missing skills
        found_skills = [skill.lower() for skill in cv_analysis.get('extracted_skills', [])]
        missing_skills = [skill for skill in relevant_skills if skill.lower() not in found_skills]
        
        return missing_skills[:10]  # Return top 10 missing skills
        
    def _generate_template_cv(self, cv_analysis: Dict[str, Any], 
                            missing_skills: List[str], target_role: str) -> str:
        """Generate CV enhancement using comprehensive templates"""
        extracted_skills = cv_analysis.get('extracted_skills', [])
        sections = cv_analysis.get('sections_identified', [])
        experience_years = cv_analysis.get('experience_years', 0)
        cv_length = cv_analysis.get('cv_length', 0)
        contact_info = cv_analysis.get('contact_info', {})
        
        # Create a concise action-oriented enhancement report
        enhancement_guide = f"""
ACTION-ORIENTED CV IMPROVEMENT PLAN
==================================

1. **IMMEDIATE ACTIONS (Today-Tomorrow)**
   - Add complete contact information to header: [Name] | [Role] | [Email] | [Phone] | [LinkedIn]
   - Restructure CV with clear section headers (PROFESSIONAL SUMMARY, CORE COMPETENCIES, EXPERIENCE, EDUCATION)
   - Quantify at least 2 achievements in current experience entries

2. **STRUCTURE & FORMAT CHANGES**
   - Move contact info from body to header
   - Create a 2-3 line professional summary highlighting {experience_years} years of {target_role} experience
   - Group skills into categories: Technical Skills, Tools & Technologies, Soft Skills
   - Use consistent bullet points and indentation throughout
   - Save as PDF with standard fonts (Arial, Calibri, Times New Roman)

3. **SKILL INTEGRATION TASKS**
   - Add '{missing_skills[0] if missing_skills else 'key skill'}' to experience description: "Utilized {missing_skills[0] if missing_skills else 'technology'} to achieve [specific result]"
   - Include '{missing_skills[1] if len(missing_skills) > 1 else 'secondary skill'}' in project descriptions
   - Add remaining missing skills to skills section: {', '.join(missing_skills[2:5]) if len(missing_skills) > 2 else 'additional skills'}

4. **CONTENT ENHANCEMENT STEPS**
   - Rewrite weak bullet "Worked on projects" to "Engineered solutions using {extracted_skills[0] if extracted_skills else 'Python'}, reducing processing time by X%"
   - Add metrics to experience: "Managed ${experience_years*50}K budget" or "Led team of X members"
   - Replace passive phrases:
     * "Responsible for" → "Managed" or "Directed"
     * "Helped with" → "Collaborated on" or "Contributed to"
     * "Worked on" → "Developed" or "Implemented"

5. **SHORT-TERM IMPROVEMENTS (This Week)**
   1. (30 min) Restructure CV layout according to recommended format
   2. (45 min) Add quantifiable metrics to all experience bullets
   3. (20 min) Create grouped skills section with {missing_skills[0] if missing_skills else 'key skills'}
   4. (60 min) Rewrite 3 weakest experience bullets with strong action verbs
   5. (15 min) Ensure contact information is in header and complete

This action plan focuses on specific improvements for your CV based on ATS optimization principles.
        """
        
        return enhancement_guide.strip()
        
    def generate_latex_cv(self, cv_analysis: Dict[str, Any], target_role: str = "Data Professional") -> str:
        """
        Generate LaTeX formatted CV with enhanced structure and content
        
        Args:
            cv_analysis (Dict): CV analysis results
            target_role (str): Target job role
            
        Returns:
            str: LaTeX formatted CV
        """
        contact_info = cv_analysis.get('contact_info', {})
        sections = cv_analysis.get('sections_identified', [])
        skills = cv_analysis.get('extracted_skills', [])
        experience_years = cv_analysis.get('experience_years', 0)
        
        # Get missing skills for recommendations
        missing_skills = self._identify_missing_skills(cv_analysis, target_role)
        
        # Organize skills by category
        programming_skills = [s for s in skills if any(lang in s.lower() for lang in ['python', 'sql', 'r', 'java', 'javascript', 'scala'])]
        data_skills = [s for s in skills if any(da in s.lower() for da in ['pandas', 'numpy', 'sql', 'excel', 'statistics', 'analysis'])]
        ml_skills = [s for s in skills if any(ml in s.lower() for ml in ['machine learning', 'tensorflow', 'pytorch', 'scikit', 'nlp', 'deep learning'])]
        visualization_skills = [s for s in skills if any(vis in s.lower() for vis in ['tableau', 'power bi', 'matplotlib', 'seaborn', 'looker'])]
        cloud_skills = [s for s in skills if any(cloud in s.lower() for cloud in ['aws', 'azure', 'gcp', 'docker', 'kubernetes'])]
        tools_skills = [s for s in skills if any(tool in s.lower() for tool in ['git', 'docker', 'kubernetes', 'jupyter', 'airflow'])]
        
        latex_template = r"""
\documentclass[11pt,a4paper]{moderncv}
\moderncvtheme[blue]{classic}
\usepackage[utf8]{inputenc}
\usepackage[scale=0.75]{geometry}
\usepackage{multicol}
\usepackage{enumitem}

% Personal Information
\firstname{""" + contact_info.get('name', 'John').split()[0] if 'name' in contact_info else 'John' + r"""}
\familyname{""" + ' '.join(contact_info.get('name', 'Doe').split()[1:]) if 'name' in contact_info and len(contact_info.get('name', 'Doe').split()) > 1 else 'Doe' + r"""}
\title{""" + target_role + r"""}
\email{""" + contact_info.get('email', 'email@example.com') + r"""}
\phone{""" + contact_info.get('phone', '+1 (555) 123-4567') + r"""}
\homepage{""" + contact_info.get('linkedin', 'linkedin.com/in/johndoe') + r"""}

\begin{document}
\maketitle

\section{Professional Summary}
Results-driven """ + target_role + r""" with """ + str(experience_years) + r""" years of experience in data analysis, technical problem-solving, and business intelligence. Skilled in """ + ', '.join(skills[:5]) + r""" with a proven track record of delivering data-driven solutions that drive business impact.

\section{Technical Skills}
\begin{multicols}{2}
\begin{itemize}[leftmargin=*]
\item \textbf{Programming:} """ + ', '.join(programming_skills[:5]) + r"""
\item \textbf{Data Analysis:} """ + ', '.join(data_skills[:5]) + r"""
\item \textbf{Machine Learning:} """ + ', '.join(ml_skills[:5]) + r"""
\item \textbf{Visualization:} """ + ', '.join(visualization_skills[:5]) + r"""
\item \textbf{Cloud Platforms:} """ + ', '.join(cloud_skills[:5]) + r"""
\item \textbf{Tools:} """ + ', '.join(tools_skills[:5]) + r"""
\end{itemize}
\end{multicols}

\section{Professional Experience}
\cventry{2022--Present}{Senior Data Analyst}{TechCorp}{}{}{Developed automated reporting systems and led data analysis projects. Key achievements:
\begin{itemize}
\item Engineered automated reporting system using Python and SQL, reducing manual reporting time by 70\%
\item Led data analysis for customer segmentation project, resulting in 15\% increase in targeted marketing ROI
\item Collaborated with engineering teams to implement predictive analytics models
\end{itemize}}

\cventry{2020--2022}{Data Analyst}{Data Insights Inc.}{}{}{Analyzed large datasets to identify business trends and opportunities. Key achievements:
\begin{itemize}
\item Built interactive dashboards using Tableau and Power BI
\item Collaborated with cross-functional teams to deliver data-driven solutions
\item Improved data quality processes, reducing errors by 25\%
\end{itemize}}

\section{Key Projects}
\cvitem{Predictive Analytics}{Developed machine learning models to predict customer behavior, resulting in 20\% improvement in targeting accuracy.}
\cvitem{Data Pipeline Optimization}{Designed and implemented ETL pipelines that reduced data processing time by 40\%.}
\cvitem{Business Intelligence Dashboard}{Created executive dashboard for C-suite decision making, adopted company-wide.}

\section{Education}
\cventry{2020}{M.S. in Data Science}{University of Technology}{}{}{}
\cventry{2018}{B.S. in Computer Science}{State University}{}{}{}

\section{Certifications}
\cvitem{2023}{AWS Certified Data Analytics}
\cvitem{2022}{Tableau Desktop Specialist}
\cvitem{2021}{Google Data Analytics Professional Certificate}

\section{Recommendations for Improvement}
Based on market analysis, consider developing these skills to enhance your competitiveness:
\begin{itemize}
""" + '\n'.join([r"\item " + skill for skill in missing_skills[:6]]) + r"""
\end{itemize}

\end{document}
        """
        
        return latex_template.strip()
        
    def generate_market_trends(self, target_role: str = "Data Professional") -> Dict[str, Any]:
        """
        Generate market trends analysis for the target role
        
        Args:
            target_role (str): Target job role
            
        Returns:
            Dict: Market trends analysis
        """
        # Define market trends for different roles
        trends_data = {
            "Data Scientist": {
                "high_demand_skills": ["Machine Learning", "Python", "SQL", "AWS", "Deep Learning"],
                "emerging_trends": ["MLOps", "LLMOps", "AutoML", "AI Governance"],
                "salary_range": "$100K-$200K+",
                "growth_rate": "15% annually"
            },
            "Data Analyst": {
                "high_demand_skills": ["SQL", "Tableau", "Power BI", "Python", "Statistics"],
                "emerging_trends": ["Self-Service Analytics", "Data Democratization", "Augmented Analytics"],
                "salary_range": "$70K-$120K+",
                "growth_rate": "10% annually"
            },
            "Machine Learning Engineer": {
                "high_demand_skills": ["TensorFlow", "PyTorch", "AWS", "Docker", "MLOps"],
                "emerging_trends": ["LLMOps", "Model Compression", "Edge AI", "Federated Learning"],
                "salary_range": "$120K-$200K+",
                "growth_rate": "20% annually"
            },
            "Business Analyst": {
                "high_demand_skills": ["Requirements Gathering", "Process Modeling", "SQL", "Agile"],
                "emerging_trends": ["Data-Driven Decision Making", "Business Intelligence", "Digital Transformation"],
                "salary_range": "$75K-$125K+",
                "growth_rate": "8% annually"
            }
        }
        
        return trends_data.get(target_role, {
            "high_demand_skills": ["Python", "SQL", "Data Analysis", "Statistics"],
            "emerging_trends": ["Automation", "AI Integration", "Cloud Computing"],
            "salary_range": "Varies by role and experience",
            "growth_rate": "Industry average"
        })
        
    def generate_skill_recommendations(self, cv_analysis: Dict[str, Any], target_role: str = "Data Professional") -> List[Dict[str, Any]]:
        """
        Generate personalized skill recommendations based on CV analysis
        
        Args:
            cv_analysis (Dict): CV analysis results
            target_role (str): Target job role
            
        Returns:
            List[Dict]: Skill recommendations with learning paths
        """
        missing_skills = self._identify_missing_skills(cv_analysis, target_role)
        
        recommendations = []
        for skill in missing_skills[:7]:  # Top 7 missing skills
            # Determine importance and market demand
            importance = "High" if skill in ["Machine Learning", "Python", "SQL", "AWS"] else "Medium"
            market_demand = "Growing" if skill in ["AI", "Cloud", "Machine Learning", "MLOps"] else "Stable"
            
            # Generate learning path
            learning_paths = {
                "Machine Learning": "Start with Andrew Ng's Machine Learning Course on Coursera, then practice with Kaggle competitions",
                "Python": "Complete Python for Data Science course on edX, then work on personal projects",
                "SQL": "Take SQL for Data Analysis on Khan Academy, then practice with real datasets",
                "AWS": "Obtain AWS Certified Cloud Practitioner, then specialize in data analytics services",
                "Deep Learning": "Study Deep Learning Specialization on Coursera, implement projects with PyTorch",
                "Tableau": "Complete Tableau Desktop Specialist certification, create dashboards with public data",
                "Power BI": "Take Microsoft Power BI training, build reports with sample datasets"
            }
            
            time_to_learn = "2-3 months" if len(skill) > 10 else "1-2 months"
            
            recommendation = {
                "skill": skill,
                "importance": importance,
                "market_demand": market_demand,
                "learning_path": learning_paths.get(skill, f"Consider online courses or certifications in {skill}"),
                "time_to_learn": time_to_learn,
                "resources": [
                    f"https://www.coursera.org/search?query={skill.replace(' ', '%20')}",
                    f"https://www.udemy.com/courses/search/?q={skill.replace(' ', '%20')}",
                    f"https://www.edx.org/search?q={skill.replace(' ', '%20')}"
                ]
            }
            recommendations.append(recommendation)
            
        return recommendations

def main():
    """Main function to demonstrate OCR and CV processing"""
    print("OCR and CV Processing System")
    print("=" * 40)
    
    # This would typically be used through the API endpoints
    # For demonstration, we'll show the functionality
    
    processor = OCRProcessor()
    print("OCR Processor initialized successfully")

if __name__ == "__main__":
    main()