import io
import json
import re
import textwrap
import time
import hashlib
from typing import List, Tuple, Dict, Any
from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# PDF and DOCX parsing
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from docx import Document
except Exception:
    Document = None

# Advanced text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception:
    nltk = None

st.set_page_config(page_title="AI Resume Cleaner Pro", page_icon="🧹", layout="wide")

# -------- Advanced Configuration --------
RESUME_TEMPLATES = {
    "Professional": {
        "sections": ["Contact", "Summary", "Experience", "Education", "Skills", "Projects"],
        "style": "clean"
    },
    "Creative": {
        "sections": ["Contact", "About", "Experience", "Education", "Skills", "Portfolio"],
        "style": "modern"
    },
    "Technical": {
        "sections": ["Contact", "Objective", "Technical Skills", "Experience", "Education", "Projects", "Certifications"],
        "style": "detailed"
    }
}

ATS_KEYWORDS = {
    "software_engineer": ["python", "java", "javascript", "react", "node.js", "sql", "git", "agile", "scrum"],
    "data_scientist": ["python", "r", "machine learning", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
    "product_manager": ["product strategy", "user research", "agile", "scrum", "analytics", "roadmap", "stakeholder"],
    "marketing": ["digital marketing", "seo", "social media", "analytics", "campaign", "brand", "content"],
    "designer": ["ui/ux", "figma", "adobe", "prototyping", "user research", "design system", "wireframing"]
}

# -------- Advanced Utilities --------
BULLET_PATTERN = re.compile(r"^\s*([\-•▪●■□♦✓✔▶»·*])\s+")

def normalize_bullets(lines: List[str]) -> List[str]:
	"""
	Normalize bullet characters and spacing so output is consistent.
	- Converts common bullet glyphs to '-'
	- Ensures single space after bullet
	"""
	normalized = []
	for line in lines:
		match = BULLET_PATTERN.match(line)
		if match:
			text = BULLET_PATTERN.sub("- ", line).rstrip()
			normalized.append(text)
		else:
			normalized.append(line.rstrip())
	return normalized


def split_into_paragraphs(text: str) -> List[str]:
	"""
	Split text into paragraphs separated by blank lines, preserving bullet blocks.
	"""
	# Normalize newlines
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	raw_paragraphs = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
	paragraphs: List[str] = []
	buffer: List[str] = []

	def flush_buffer():
		if buffer:
			paragraphs.append("\n".join(buffer))
			buffer.clear()

	for para in raw_paragraphs:
		lines = [ln.rstrip() for ln in para.split("\n")]
		# If many lines start with bullet, keep as one paragraph
		bullet_count = sum(1 for ln in lines if BULLET_PATTERN.match(ln))
		if bullet_count >= max(2, len(lines) // 2):
			flush_buffer()
			paragraphs.append("\n".join(lines))
		else:
			buffer.extend(lines)
			flush_buffer()

	return paragraphs if paragraphs else [text]


def analyze_resume_structure(text: str) -> Dict[str, Any]:
    """
    Analyze resume structure and provide insights.
    """
    analysis = {
        "word_count": len(text.split()),
        "sections": [],
        "has_contact": False,
        "has_experience": False,
        "has_education": False,
        "has_skills": False,
        "bullet_points": 0,
        "action_verbs": 0,
        "quantified_achievements": 0
    }
    
    # Detect sections
    section_keywords = {
        "contact": ["contact", "email", "phone", "address", "linkedin"],
        "summary": ["summary", "objective", "profile", "about"],
        "experience": ["experience", "work history", "employment", "career"],
        "education": ["education", "academic", "degree", "university", "college"],
        "skills": ["skills", "technical skills", "competencies", "expertise"],
        "projects": ["projects", "portfolio", "work samples"],
        "certifications": ["certifications", "certificates", "licenses"]
    }
    
    text_lower = text.lower()
    for section, keywords in section_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            analysis["sections"].append(section)
            if section in ["contact"]:
                analysis["has_contact"] = True
            elif section in ["experience"]:
                analysis["has_experience"] = True
            elif section in ["education"]:
                analysis["has_education"] = True
            elif section in ["skills"]:
                analysis["has_skills"] = True
    
    # Count bullet points
    analysis["bullet_points"] = len(re.findall(r'^\s*[-•▪●■□♦✓✔▶»·*]', text, re.MULTILINE))
    
    # Count action verbs
    action_verbs = ["developed", "created", "implemented", "designed", "managed", "led", "improved", "optimized", "built", "launched"]
    analysis["action_verbs"] = sum(1 for verb in action_verbs if verb in text_lower)
    
    # Count quantified achievements
    analysis["quantified_achievements"] = len(re.findall(r'\d+%|\d+\+|\d+x|\$\d+', text))
    
    return analysis

def calculate_ats_score(text: str, job_title: str = "software_engineer") -> Dict[str, Any]:
    """
    Calculate ATS (Applicant Tracking System) compatibility score.
    """
    if job_title not in ATS_KEYWORDS:
        job_title = "software_engineer"
    
    target_keywords = ATS_KEYWORDS[job_title]
    text_lower = text.lower()
    
    found_keywords = []
    missing_keywords = []
    
    for keyword in target_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    score = (len(found_keywords) / len(target_keywords)) * 100
    
    return {
        "score": round(score, 1),
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "total_keywords": len(target_keywords),
        "matched_count": len(found_keywords)
    }

def enhance_content_with_ai(text: str) -> str:
    """
    Enhance resume content with AI-powered suggestions.
    """
    enhanced = text
    
    # Replace weak action verbs with stronger ones
    verb_replacements = {
        "did": "accomplished",
        "made": "created",
        "got": "achieved",
        "helped": "contributed to",
        "worked on": "developed",
        "used": "utilized",
        "put": "implemented",
        "fixed": "resolved",
        "changed": "optimized",
        "started": "initiated"
    }
    
    for weak, strong in verb_replacements.items():
        enhanced = re.sub(r'\b' + weak + r'\b', strong, enhanced, flags=re.IGNORECASE)
    
    # Add quantifiers where missing
    enhanced = re.sub(r'\bimproved\b', 'significantly improved', enhanced, flags=re.IGNORECASE)
    enhanced = re.sub(r'\bincreased\b', 'dramatically increased', enhanced, flags=re.IGNORECASE)
    enhanced = re.sub(r'\breduced\b', 'substantially reduced', enhanced, flags=re.IGNORECASE)
    
    return enhanced

def apply_languagetool(text: str, api_url: str = "https://api.languagetool.org/v2/check", language: str = "en-US") -> Tuple[str, List[dict]]:
	"""
	Call LanguageTool public API to get grammar/punctuation/style suggestions and apply safe replacements.
	Returns (corrected_text, matches_metadata).
	"""
	if not text.strip():
		return text, []
	payload = {
		"text": text,
		"language": language,
		"enabledRules": "",  # keep default
		"enabledCategories": "",  # keep default
		"level": "default",
	}
	try:
		response = requests.post(api_url, data=payload, timeout=30)
		response.raise_for_status()
		data = response.json()
	except Exception:
		# If API is unavailable, return original text
		return text, []

	corrected = text
	matches = data.get("matches", [])
	# Apply replacements from end to start to keep offsets valid
	# Prefer the first replacement if available
	for m in sorted(matches, key=lambda x: x.get("offset", 0), reverse=True):
		offset = m.get("offset", 0)
		length = m.get("length", 0)
		reps = m.get("replacements", [])
		if length < 0 or offset < 0:
			continue
		if offset + length > len(corrected):
			continue
		replacement = None
		if reps:
			replacement = reps[0].get("value")
		# If no replacement, skip
		if replacement is None:
			continue
		corrected = corrected[:offset] + replacement + corrected[offset + length :]
	return corrected, matches


def clean_text_preserving_layout(raw_text: str) -> str:
    """
    Clean grammar/punctuation while preserving layout and bullets.
    Strategy:
    - Remove PDF artifacts and formatting issues
    - Normalize bullets and spacing
    - Split into paragraphs (single text or bullet blocks)
    - Send each paragraph to LanguageTool to avoid position drift across blocks
    - Rebuild with blank lines between paragraphs
    """
    if not raw_text.strip():
        return raw_text

    # Remove PDF artifacts and common formatting issues
    cleaned = raw_text
    # Remove PDF character IDs like (cid:132)
    cleaned = re.sub(r'\(cid:\d+\)', '', cleaned)
    # Fix common spacing issues
    cleaned = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)  # Add space between camelCase
    cleaned = re.sub(r'([a-z])(\d)', r'\1 \2', cleaned)  # Add space between letter and number
    cleaned = re.sub(r'(\d)([A-Z])', r'\1 \2', cleaned)  # Add space between number and letter
    # Fix common grammar issues
    cleaned = re.sub(r'\b([a-z]+)are\b', r'\1 are', cleaned)  # Fix "Developedareal" -> "Developed a real"
    cleaned = re.sub(r'\b([a-z]+)an\b', r'\1 an', cleaned)  # Fix "Deployedan" -> "Deployed an"
    cleaned = re.sub(r'\b([a-z]+)with\b', r'\1 with', cleaned)  # Fix "Integratedwith" -> "Integrated with"
    cleaned = re.sub(r'\b([a-z]+)for\b', r'\1 for', cleaned)  # Fix "Designedfor" -> "Designed for"
    cleaned = re.sub(r'\b([a-z]+)to\b', r'\1 to', cleaned)  # Fix "Implementedto" -> "Implemented to"
    cleaned = re.sub(r'\b([a-z]+)in\b', r'\1 in', cleaned)  # Fix "Specializedin" -> "Specialized in"
    # Fix common punctuation issues
    cleaned = re.sub(r'([a-z])([A-Z][a-z])', r'\1. \2', cleaned)  # Add period before new sentences
    cleaned = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned)  # Ensure space after punctuation
    # Clean up multiple spaces
    cleaned = re.sub(r' +', ' ', cleaned)
    # Clean up multiple newlines
    cleaned = re.sub(r'\n+', '\n', cleaned)

    lines = cleaned.replace("\t", "    ").split("\n")
    lines = normalize_bullets(lines)
    normalized_text = "\n".join(lines)

    paragraphs = split_into_paragraphs(normalized_text)

    cleaned_paragraphs: List[str] = []
    for para in paragraphs:
        # Keep bullet structure line-by-line
        if any(BULLET_PATTERN.match(ln) for ln in para.split("\n")):
            cleaned_lines = []
            for ln in para.split("\n"):
                if BULLET_PATTERN.match(ln):
                    # Clean only the text after the bullet
                    bullet_body = BULLET_PATTERN.sub("", ln, count=1)
                    fixed, _ = apply_languagetool(bullet_body)
                    cleaned_lines.append("- " + fixed.strip())
                else:
                    fixed, _ = apply_languagetool(ln)
                    cleaned_lines.append(fixed)
            cleaned_paragraphs.append("\n".join(cleaned_lines))
        else:
            fixed, _ = apply_languagetool(para)
            cleaned_paragraphs.append(fixed)

    cleaned_text = ("\n\n").join(p.strip() for p in cleaned_paragraphs if p is not None)
    # Collapse excessive blank lines
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def read_txt(file_bytes: bytes, encoding_fallbacks: List[str] = ["utf-8", "utf-16", "latin-1"]) -> str:
	for enc in encoding_fallbacks:
		try:
			return file_bytes.decode(enc)
		except Exception:
			continue
	return file_bytes.decode("utf-8", errors="ignore")


def read_pdf(file_bytes: bytes) -> str:
	if pdfplumber is None:
		raise RuntimeError("pdfplumber not installed. Please install requirements.")
	text_parts: List[str] = []
	with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
		for page in pdf.pages:
			extracted = page.extract_text() or ""
			text_parts.append(extracted)
	return "\n".join(text_parts)


def read_docx(file_bytes: bytes) -> str:
	if Document is None:
		raise RuntimeError("python-docx not installed. Please install requirements.")
	f = io.BytesIO(file_bytes)
	doc = Document(f)
	lines: List[str] = []
	for p in doc.paragraphs:
		txt = p.text or ""
		# Preserve simple bullets if they exist
		if p.style and p.style.name and "List" in p.style.name:
			if not BULLET_PATTERN.match(txt):
				txt = f"- {txt}"
		lines.append(txt)
	return "\n".join(lines)


def write_docx_from_text(text: str) -> bytes:
	"""
	Build a simple DOCX from plain text. Bullet lines (starting with '-') become bullet list items.
	"""
	if Document is None:
		raise RuntimeError("python-docx not installed. Please install requirements.")
	doc = Document()
	for block in text.split("\n"):
		if BULLET_PATTERN.match(block) or block.strip().startswith("- "):
			# Bullet paragraph
			para = doc.add_paragraph(block.replace("- ", "", 1).strip(), style="List Bullet")
		elif block.strip() == "":
			doc.add_paragraph("")  # blank line
		else:
			doc.add_paragraph(block)
	out = io.BytesIO()
	doc.save(out)
	return out.getvalue()


def extract_text(file_name: str, file_bytes: bytes) -> Tuple[str, str]:
	"""
	Extract raw text and return (text, detected_type).
	"""
	name_lower = file_name.lower()
	if name_lower.endswith(".txt"):
		return read_txt(file_bytes), "txt"
	if name_lower.endswith(".pdf"):
		return read_pdf(file_bytes), "pdf"
	if name_lower.endswith(".docx"):
		return read_docx(file_bytes), "docx"
	# Fallback: try decoding as text
	try:
		return read_txt(file_bytes), "txt"
	except Exception:
		raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")

def process_resume(raw_text: str, lang: str, enable_ai: bool, enable_grammar: bool, job_title: str):
    """Process resume with all selected enhancements."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Basic cleaning
    status_text.text("🧹 Basic cleaning and formatting...")
    progress_bar.progress(20)
    cleaned = clean_text_preserving_layout(raw_text)
    
    # Step 2: AI enhancement
    if enable_ai:
        status_text.text("🤖 AI content enhancement...")
        progress_bar.progress(40)
        cleaned = enhance_content_with_ai(cleaned)
    
    # Step 3: Grammar check
    if enable_grammar:
        status_text.text("📝 Advanced grammar checking...")
        progress_bar.progress(60)
        # Apply grammar check to cleaned text
        cleaned, _ = apply_languagetool(cleaned, language=lang)
    
    # Step 4: Final formatting
    status_text.text("✨ Final formatting and optimization...")
    progress_bar.progress(80)
    
    # Step 5: Complete
    status_text.text("✅ Processing complete!")
    progress_bar.progress(100)
    
    # Display results
    st.subheader("🎉 Enhanced Resume")
    
    # Side-by-side comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before**")
        st.text_area("Original", raw_text, height=300, key="original")
    
    with col2:
        st.markdown("**After**")
        st.text_area("Enhanced", cleaned, height=300, key="enhanced")
    
    # Download options
    st.subheader("📥 Download Options")
    
    # Prepare downloads
    cleaned_txt = cleaned.encode("utf-8")
    try:
        cleaned_docx = write_docx_from_text(cleaned)
    except Exception as e:
        cleaned_docx = None
        st.info(f"DOCX generation note: {e}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📄 Download as TXT",
            data=cleaned_txt,
            file_name="enhanced_resume.txt",
            mime="text/plain",
        )
    with col2:
        if cleaned_docx:
            st.download_button(
                label="📝 Download as DOCX",
                data=cleaned_docx,
                file_name="enhanced_resume.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
    with col3:
        # Generate PDF (placeholder)
        st.download_button(
            label="📋 Download as PDF",
            data=cleaned_txt,  # Placeholder
            file_name="enhanced_resume.pdf",
            mime="application/pdf",
            disabled=True,
            help="PDF generation coming soon!"
        )

def generate_template(template_name: str, raw_text: str):
    """Generate resume template based on selected style."""
    template = RESUME_TEMPLATES[template_name]
    
    st.subheader(f"📋 {template_name} Template")
    st.info(f"Template style: {template['style']}")
    
    # Show template structure
    st.markdown("**Template Sections:**")
    for i, section in enumerate(template["sections"], 1):
        st.markdown(f"{i}. {section}")
    
    # Generate template content
    template_content = f"# {template_name} Resume Template\n\n"
    for section in template["sections"]:
        template_content += f"## {section}\n\n"
        if section == "Contact":
            template_content += "Name: [Your Name]\nEmail: [your.email@example.com]\nPhone: [Your Phone]\nLinkedIn: [Your LinkedIn]\n\n"
        elif section == "Summary":
            template_content += "Professional summary highlighting your key achievements and career objectives.\n\n"
        elif section == "Experience":
            template_content += "### [Job Title] - [Company Name]\n[Start Date] - [End Date]\n- [Achievement 1]\n- [Achievement 2]\n- [Achievement 3]\n\n"
        elif section == "Education":
            template_content += "### [Degree] - [University Name]\n[Start Date] - [End Date]\nGPA: [Your GPA]\n\n"
        elif section == "Skills":
            template_content += "- [Technical Skill 1]\n- [Technical Skill 2]\n- [Soft Skill 1]\n- [Soft Skill 2]\n\n"
    
    st.text_area("Template Content", template_content, height=400)
    
    # Download template
    template_bytes = template_content.encode("utf-8")
    st.download_button(
        label="📥 Download Template",
        data=template_bytes,
        file_name=f"{template_name.lower()}_resume_template.txt",
        mime="text/plain",
    )


# -------- Advanced UI --------
st.title("🧹 AI Resume Cleaner Pro")
st.markdown("**Advanced resume optimization with AI-powered analysis, ATS scoring, and content enhancement**")

# Sidebar with advanced options
with st.sidebar:
    st.header("⚙️ Advanced Options")
    
    # Language selection
    lang = st.selectbox("Language", ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE"], index=0)
    
    # Job title for ATS optimization
    job_title = st.selectbox("Target Job Role", list(ATS_KEYWORDS.keys()), index=0)
    
    # Resume template
    template = st.selectbox("Resume Template", list(RESUME_TEMPLATES.keys()), index=0)
    
    # Enhancement options
    st.subheader("🔧 Enhancement Options")
    enable_ai_enhancement = st.checkbox("AI Content Enhancement", value=True)
    enable_ats_optimization = st.checkbox("ATS Optimization", value=True)
    enable_grammar_check = st.checkbox("Advanced Grammar Check", value=True)
    enable_structure_analysis = st.checkbox("Structure Analysis", value=True)
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        max_file_size = st.slider("Max File Size (MB)", 1, 10, 5)
        processing_mode = st.radio("Processing Mode", ["Fast", "Thorough", "Custom"])
        
        if processing_mode == "Custom":
            custom_enhancements = st.multiselect(
                "Custom Enhancements",
                ["Action Verbs", "Quantifiers", "Keywords", "Formatting", "Bullet Points"]
            )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader(
        "📁 Upload your resume file", 
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT (Max 5MB)"
    )

with col2:
    st.markdown("### 📊 Quick Stats")
    if uploaded:
        file_size = len(uploaded.getvalue()) / 1024 / 1024
        st.metric("File Size", f"{file_size:.2f} MB")
        st.metric("File Type", uploaded.name.split('.')[-1].upper())
    else:
        st.info("Upload a file to see stats")

if uploaded is not None:
    file_bytes = uploaded.read()
    
    # Check file size
    file_size_mb = len(file_bytes) / 1024 / 1024
    if file_size_mb > max_file_size:
        st.error(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({max_file_size} MB)")
        st.stop()

    with st.spinner("🔍 Extracting and analyzing text..."):
        try:
            raw_text, detected = extract_text(uploaded.name, file_bytes)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

    if not raw_text.strip():
        st.warning("We couldn't extract readable text from this file.")
        st.stop()

    # Advanced analysis
    if enable_structure_analysis:
        with st.spinner("📊 Analyzing resume structure..."):
            analysis = analyze_resume_structure(raw_text)
            ats_analysis = calculate_ats_score(raw_text, job_title)

    # Display analysis results
    if enable_structure_analysis:
        st.subheader("📊 Resume Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Word Count", analysis["word_count"])
        with col2:
            st.metric("Sections Found", len(analysis["sections"]))
        with col3:
            st.metric("Bullet Points", analysis["bullet_points"])
        with col4:
            st.metric("Action Verbs", analysis["action_verbs"])

        # ATS Score
        if enable_ats_optimization:
            st.subheader("🎯 ATS Compatibility Score")
            ats_score = ats_analysis["score"]
            
            # Color-coded score
            if ats_score >= 80:
                score_color = "green"
            elif ats_score >= 60:
                score_color = "orange"
            else:
                score_color = "red"
            
            st.markdown(f"**Score: <span style='color: {score_color}'>{ats_score}%</span>**", unsafe_allow_html=True)
            
            # Progress bar
            st.progress(ats_score / 100)
            
            # Keywords analysis
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Found Keywords: {', '.join(ats_analysis['found_keywords'][:5])}")
            with col2:
                if ats_analysis['missing_keywords']:
                    st.warning(f"❌ Missing: {', '.join(ats_analysis['missing_keywords'][:5])}")

    # Original text display
    with st.expander("📄 Original Extracted Text", expanded=False):
        st.text_area("Original", raw_text, height=200, help="This is the extracted text used for cleaning.")

    # Processing options
    st.subheader("🚀 Processing Options")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✨ Clean & Enhance Resume", type="primary"):
            process_resume(raw_text, lang, enable_ai_enhancement, enable_grammar_check, job_title)
    
    with col2:
        if st.button("📋 Generate Template"):
            generate_template(template, raw_text)

else:
    st.info("📁 Choose a PDF, DOCX, or TXT file to get started.")
    st.caption("💡 **Pro Tips**: Use the sidebar to customize language, job role, and enhancement options for better results.")



# Footer
st.markdown("---")
st.caption("Built with Streamlit + LanguageTool. No data stored. Works offline except for grammar API calls.")


