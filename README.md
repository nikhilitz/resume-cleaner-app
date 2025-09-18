# AI Resume Cleaner 🧹

[![Live App](https://img.shields.io/badge/🚀_Live_App-Streamlit-blue?style=for-the-badge&logo=streamlit)](https://resume-cleaner-app-exupajpkpz8p9bhb5dbmos.streamlit.app)

A simple web app that cleans up your resume by fixing grammar, punctuation, and formatting issues while preserving the original content and meaning.

## 🌐 Live Demo

**Try it now**: [https://resume-cleaner-app-exupajpkpz8p9bhb5dbmos.streamlit.app](https://resume-cleaner-app-exupajpkpz8p9bhb5dbmos.streamlit.app)

## Features

- **Multi-format support**: Upload PDF, DOCX, or TXT files
- **Smart cleaning**: Fixes grammar, punctuation, and formatting issues
- **Bullet point preservation**: Maintains consistent bullet formatting
- **Download options**: Get cleaned resume as TXT or DOCX
- **Language support**: English (US/UK) via LanguageTool API

## How to Use

1. Upload your resume file (PDF, DOCX, or TXT)
2. Click "Clean Resume ✨" to process the text
3. Review the cleaned version
4. Download as TXT or DOCX

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
# Clone the repository
git clone <your-github-repo-url>
cd ai-resume-cleaner

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## Deployment

This app is ready for deployment on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push this code to a GitHub repository
2. Sign up at [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Deploy from your repository
5. Your app will be live at `https://username-appname.streamlit.app`

## Technical Details

- **Framework**: Streamlit
- **Text Processing**: LanguageTool Public API
- **File Parsing**: pdfplumber (PDF), python-docx (DOCX)
- **Language**: Python 3.9+

## Privacy

- No data is stored on our servers
- Files are processed in memory only
- LanguageTool API calls are made for grammar checking
- All processing happens in your browser session

## License

MIT License - feel free to use and modify as needed.

## Contributing

Pull requests are welcome! For major changes, please open an issue first.
