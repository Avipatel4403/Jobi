# Jobi - LLM CLI Tool for Professional Writing

A local CLI tool that combines your personal context (resume, work history, skills) with company information to generate professional writing like cover letters and cold emails using local Ollama models and a RAG system.

## 🚀 Features

- **Local RAG System**: Uses ChromaDB to store and retrieve your personal profile data
- **Local LLM Integration**: Powered by Ollama (no API keys required for chat)
- **Web Search Integration**: Search the web using Ollama's web search API
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

5. **(Optional) Set up web search** - If you want to use web search features:
   ```bash
   # Copy the example .env file
   cp .env.example .env
   
   # Edit .env and add your API key from https://ollama.com/settings/keys
   # OLLAMA_API_KEY=your-actual-api-key-here
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

### 6. Web Search (Requires Ollama API Key)

Search the web using Ollama's web search API:

```bash
# Method 1: Create a .env file (RECOMMENDED)
# Create a .env file in the project root with your API key
echo 'OLLAMA_API_KEY=your-api-key-here' > .env

# The key will be automatically loaded when you run commands
poetry run jobi websearch "What is Ollama?"

# Method 2: Export as environment variable (session only)
export OLLAMA_API_KEY='your-api-key-here'
poetry run jobi websearch "What is Ollama?"

# Method 3: Pass directly to command
poetry run jobi websearch "What is Ollama?" --api-key "your-api-key"
```

**Get your API key:**
1. Create a free account at [https://ollama.com](https://ollama.com)
2. Get your API key from [https://ollama.com/settings/keys](https://ollama.com/settings/keys)
3. Add it to `.env` file in the project root

**Examples:**

```bash
# Search the web
poetry run jobi websearch "Python frameworks" --max-results 3

# Get JSON output
poetry run jobi websearch "AI news" --json-output

# Fetch content from a specific URL
poetry run jobi webfetch "https://ollama.com"

# Show all links found on the page
poetry run jobi webfetch "https://ollama.com" --show-links
```

**Using in Python scripts:**

```python
from jobi.web import OllamaWebSearch

# API key is automatically loaded from .env file or environment
# No need to pass it explicitly!
web_search = OllamaWebSearch()

# Search the web
results = web_search.web_search("What is Ollama?", max_results=5)
for result in results:
    print(f"{result.title}: {result.url}")
    print(f"{result.content}\n")

# Fetch a specific URL
page = web_search.web_fetch("https://ollama.com")
print(f"Title: {page.title}")
print(f"Content: {page.content}")
print(f"Links: {page.links}")

# Or use convenience functions
from jobi.web import web_search, web_fetch

results = web_search("Python programming", max_results=3)
page = web_fetch("https://example.com")
```

See `examples/web_search_examples.py` and `examples/search_agent_example.py` for more detailed examples.

### 7. Manage Documents

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
│   ├── cli.py              # Main CLI interface
│   ├── rag/                # RAG system modules
│   │   ├── core.py         # ChromaDB RAG system
│   │   ├── chunkers.py     # Document chunking strategies
│   │   ├── ingestion.py    # Document ingestion
│   │   └── utils.py        # Utility functions
│   ├── web/                # Web search module
│   │   ├── __init__.py     # Module exports
│   │   ├── models.py       # Data models
│   │   └── client.py       # API client
│   ├── ollama_client.py    # Ollama integration
│   └── chat.py             # Chat handler and generation logic
├── docs/
│   └── WEB_SEARCH.md       # Web search documentation
├── examples/
│   ├── web_search_examples.py      # Web search examples
│   ├── search_agent_example.py     # Search agent with LLM
│   └── custom_chunking_examples.py # Chunking examples
├── data/
│   └── chromadb/           # ChromaDB persistence
├── outputs/                # Generated content saved here
├── .env                    # API keys (git-ignored)
└── pyproject.toml          # Poetry configuration
```

## 🔧 Configuration

### Ollama API Key (for Web Search)

To use the web search features, you need an Ollama API key:

1. Create a free account at [https://ollama.com](https://ollama.com)
2. Get your API key at [https://ollama.com/settings/keys](https://ollama.com/settings/keys)
3. **Add to `.env` file** (recommended):

```bash
# Create .env file in project root
echo 'OLLAMA_API_KEY=your-actual-api-key-here' > .env
```

The `.env` file is already in `.gitignore`, so your API key will be safe and won't be committed to git.

**Alternative methods:**
- Export in shell (temporary): `export OLLAMA_API_KEY='your-key'`
- Pass to command: `jobi websearch "query" --api-key "your-key"`
- Add to shell profile for persistence: `echo 'export OLLAMA_API_KEY="your-key"' >> ~/.zshrc`

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

ChromaDB data is stored in `./data/chromadb/` directory. This persists your profile data between sessions.

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

# Web search examples (requires OLLAMA_API_KEY in .env file)
poetry run jobi websearch "Latest AI trends 2025"
poetry run jobi websearch "Python async programming" --max-results 5
poetry run jobi webfetch "https://docs.python.org" --show-links

# Run example scripts
python examples/web_search_examples.py
python examples/search_agent_example.py

# Management
poetry run jobi remove old_resume.txt
poetry run jobi list --verbose
poetry run jobi stats
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


## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Ollama](https://ollama.ai/) for local LLM serving and web search API
- [Click](https://click.palletsprojects.com/) for the CLI framework

---

## 📚 Additional Resources

- [Ollama Documentation](https://docs.ollama.com/)
- [Ollama Web Search API](https://docs.ollama.com/web-search)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Examples Directory](./examples/) - See working code examples
