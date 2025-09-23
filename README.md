# Jobi - LLM CLI Tool for Professional Writing

A local CLI tool that combines your personal context (resume, work history, skills) with company information to generate professional writing like cover letters and cold emails using local Ollama models and a RAG system.

## 🚀 Features

- **Local RAG System**: Uses ChromaDB to store and retrieve your personal profile data
- **Local LLM Integration**: Powered by Ollama (no API keys required)
- **Smart Context Retrieval**: Automatically finds relevant information from your profile
- **Professional Writing**: Generates tailored cover letters, cold emails, and application responses
- **Automatic Saving**: Saves all generated content with organized naming
- **Web Scraping**: Can extract company information from URLs
- **CLI Interface**: Easy-to-use command-line interface

## 📋 Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally
- Poetry (for dependency management)

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd jobi
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Start Ollama** (in another terminal):
   ```bash
   ollama serve
   ```

4. **Pull a model** (if you haven't already):
   ```bash
   ollama pull gemma3:latest
   # or
   ollama pull llama3.2
   ```

## 📖 Usage

### 1. Ingest Your Profile Data

First, add your personal documents (resume, cover letters, work history, etc.) to the RAG system:

```bash
# Ingest your resume
poetry run jobi ingest resume.txt

# Ingest cover letter examples
poetry run jobi ingest cover_letters.txt

# Add metadata (optional)
poetry run jobi ingest work_experience.txt -m type=experience -m category=technical
```

### 2. List Your Documents

See what's in your profile:

```bash
# Simple list
poetry run jobi list

# Detailed view
poetry run jobi list --verbose
```

### 3. Search Your Profile

Test what context will be retrieved:

```bash
poetry run jobi search "Python machine learning experience"
```

### 4. Generate Professional Writing

Start an interactive chat session:

```bash
poetry run jobi chat
```

You'll be prompted for:
- Company name or URL
- What you want to write (cover letter, cold email, etc.)

The tool will:
1. Retrieve relevant context from your profile
2. Optionally scrape company information from their website
3. Generate personalized content using Ollama
4. Display the result and save it to `./outputs/`

### 5. Advanced Options

```bash
# Use a specific model
poetry run jobi chat --model llama3.2

# Provide company and query upfront
poetry run jobi chat --company "Google" --query "Write a cover letter for SWE role"

# Adjust context retrieval
poetry run jobi chat --context-limit 10
```

### 6. Manage Documents

```bash
# Remove a document
poetry run jobi remove resume.txt

# Skip confirmation
poetry run jobi remove old_resume.txt --confirm
```

## 📁 Project Structure

```
jobi/
├── jobi/
│   ├── __init__.py
│   ├── cli.py          # Main CLI interface
│   ├── rag.py          # ChromaDB RAG system
│   ├── ollama_client.py # Ollama integration
│   └── chat.py         # Chat handler and generation logic
├── outputs/            # Generated content saved here
├── chroma_db/          # ChromaDB persistence (created automatically)
├── sample_resume.txt   # Example resume file
└── pyproject.toml      # Poetry configuration
```

## 🔧 Configuration

### Model Selection

Available models depend on what you have installed with Ollama:

```bash
# List available models
ollama list

# Pull additional models
ollama pull gemma3:latest
ollama pull llama3.2
ollama pull mistral
```

### ChromaDB Storage

ChromaDB data is stored in `./chroma_db/` directory. This persists your profile data between sessions.

## 📝 Example Workflow

1. **Setup your profile**:
   ```bash
   poetry run jobi ingest resume.txt
   poetry run jobi ingest cover_letter_examples.txt
   poetry run jobi ingest project_descriptions.txt
   ```

2. **Verify your data**:
   ```bash
   poetry run jobi list -v
   poetry run jobi search "software engineering experience"
   ```

3. **Generate content**:
   ```bash
   poetry run jobi chat
   ```
   - Enter: "Google" (company)
   - Enter: "Write a cover letter for Senior Software Engineer position"
   - Review generated content
   - Find saved file in `./outputs/`

## 🎯 Tips for Best Results

1. **Quality Input Data**: Add comprehensive profile documents (resume, project descriptions, work samples)

2. **Specific Queries**: Be specific about what you want ("cover letter for ML engineer role" vs "write something")

3. **Company Research**: Provide URLs when possible for better company context

4. **Iterate**: Use the regeneration option to try different approaches

5. **Review Output**: Always review and customize the generated content before use

## 🔍 Example Commands

```bash
# Full workflow example
poetry run jobi ingest resume.txt
poetry run jobi ingest portfolio_projects.txt
poetry run jobi list
poetry run jobi chat --company "https://openai.com" --query "Write a cold email for ML Engineer position"

# Search and test
poetry run jobi search "machine learning projects"
poetry run jobi search "leadership experience"

# Management
poetry run jobi remove old_resume.txt
poetry run jobi list --verbose
```

## 🚨 Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

### Model Not Found
```bash
# List available models
ollama list

# Pull the model you want to use
ollama pull gemma3:latest
```

### ChromaDB Issues
- Delete `./chroma_db/` folder to reset the database
- Re-ingest your documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Ollama](https://ollama.ai/) for local LLM serving
- [Click](https://click.palletsprojects.com/) for the CLI framework
