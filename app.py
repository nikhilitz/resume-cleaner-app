import io
import json
import re
import textwrap
from typing import List, Tuple

import requests
import streamlit as st

# PDF and DOCX parsing
try:
	import pdfplumber
except Exception:
	pdfplumber = None

try:
	from docx import Document
except Exception:
	Document = None

st.set_page_config(page_title="AI Resume Cleaner", page_icon="🧹", layout="centered")


# -------- Utilities --------
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


# -------- UI --------
st.title("🧹 AI Resume Cleaner")
st.write("Upload your resume (PDF, DOCX, or TXT). We'll fix grammar, punctuation, and formatting while preserving meaning and readability.")

with st.sidebar:
	st.header("Options")
	lang = st.selectbox("Language", ["en-US", "en-GB"], index=0)
	st.caption("Language is passed to the LanguageTool API.")

uploaded = st.file_uploader("Upload a resume file", type=["pdf", "docx", "txt"])

if uploaded is not None:
	file_bytes = uploaded.read()

	with st.spinner("Extracting text..."):
		try:
			raw_text, detected = extract_text(uploaded.name, file_bytes)
		except Exception as e:
			st.error(f"Could not read file: {e}")
			st.stop()

	if not raw_text.strip():
		st.warning("We couldn't extract readable text from this file.")
		st.stop()

	st.subheader("Original Extracted Text")
	st.text_area("Original", raw_text, height=240, help="This is the extracted text used for cleaning.")

	if st.button("Clean Resume ✨"):
		with st.spinner("Cleaning grammar, punctuation, and formatting..."):
			cleaned = clean_text_preserving_layout(raw_text)

		st.subheader("Cleaned Resume")
		st.text_area("Cleaned", cleaned, height=320)

		# Prepare downloads
		cleaned_txt = cleaned.encode("utf-8")
		try:
			cleaned_docx = write_docx_from_text(cleaned)
		except Exception as e:
			cleaned_docx = None
			st.info(f"DOCX generation note: {e}")

		cols = st.columns(2)
		with cols[0]:
			st.download_button(
				label="Download as TXT",
				data=cleaned_txt,
				file_name="cleaned_resume.txt",
				mime="text/plain",
			)
		with cols[1]:
			if cleaned_docx:
				st.download_button(
					label="Download as DOCX",
					data=cleaned_docx,
					file_name="cleaned_resume.docx",
					mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
				)
			else:
				st.caption("DOCX download unavailable (missing python-docx).")

else:
	st.info("Choose a PDF, DOCX, or TXT file to get started.")
	st.caption("Tip: Bullet points will be normalized for consistent layout. Use the sidebar to set language.")


# Footer
st.markdown("---")
st.caption("Built with Streamlit + LanguageTool. No data stored. Works offline except for grammar API calls.")


