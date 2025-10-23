"""CLI interface for Jobi - LLM CLI tool for professional writing"""

import click
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .rag import RAGSystem, DefaultChunker, SemanticChunker, DocumentTypeChunker
from .ollama_client import OllamaClient
from .chat import ChatHandler
from .web import OllamaWebSearch


@click.group()
@click.version_option()
@click.pass_context
def main(ctx):
    """Jobi - LLM CLI tool for professional writing with RAG
    
    A tool that combines your personal context (resume, work history) with 
    company information to generate professional writing like cover letters 
    and cold emails using local Ollama models.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Initialize RAG system with default chunker
    try:
        ctx.obj['rag'] = RAGSystem(chunker=DocumentTypeChunker())
    except Exception as e:
        click.echo(f"Error initializing RAG system: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('filepath', type=click.Path(exists=True, readable=True))
@click.option('--metadata', '-m', help='Additional metadata as key=value pairs', multiple=True)
@click.pass_context
def ingest(ctx, filepath: str, metadata: tuple):
    """Ingest a document into the RAG system.
    
    FILEPATH: Path to the text file to ingest (resume, cover letters, notes, etc.)
    """
    rag_system = ctx.obj['rag']
    
    # Parse metadata
    doc_metadata = {}
    for item in metadata:
        if '=' in item:
            key, value = item.split('=', 1)
            doc_metadata[key.strip()] = value.strip()
    
    click.echo(f"Ingesting document: {filepath}")
    
    if doc_metadata:
        click.echo(f"With metadata: {doc_metadata}")
    
    success = rag_system.ingest_document(filepath, doc_metadata)
    
    if success:
        click.echo(f"✅ Successfully ingested {Path(filepath).name}")
        
        # Show stats
        stats = rag_system.get_collection_stats()
        click.echo(f"📊 Collection now has {stats.get('total_documents', 0)} documents with {stats.get('total_chunks', 0)} chunks")
    else:
        click.echo("❌ Failed to ingest document", err=True)
        sys.exit(1)


@main.command()
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
@click.pass_context
def list(ctx, verbose: bool):
    """List all documents in the RAG system."""
    rag_system = ctx.obj['rag']
    
    documents = rag_system.list_documents()
    
    if not documents:
        click.echo("No documents found in the RAG system.")
        click.echo("Use 'jobi ingest <file>' to add documents.")
        return
    
    # Show collection stats
    stats = rag_system.get_collection_stats()
    click.echo(f"📊 Collection Statistics:")
    click.echo(f"   Documents: {stats.get('total_documents', 0)}")
    click.echo(f"   Chunks: {stats.get('total_chunks', 0)}")
    click.echo(f"   Total characters: {stats.get('total_characters', 0):,}")
    click.echo()
    
    # List documents
    click.echo("📄 Documents:")
    for doc in documents:
        filename = doc.get('filename', 'Unknown')
        chunk_count = doc.get('chunk_count', 0)
        total_size = doc.get('total_size', 0)
        
        click.echo(f"   • {filename}")
        
        if verbose:
            filepath = doc.get('filepath', 'Unknown')
            click.echo(f"     Path: {filepath}")
            click.echo(f"     Chunks: {chunk_count}")
            click.echo(f"     Size: {total_size:,} characters")
            click.echo()


@main.command()
@click.argument('filename')
@click.option('--confirm', '-y', is_flag=True, help='Skip confirmation prompt')
@click.pass_context
def remove(ctx, filename: str, confirm: bool):
    """Remove a document from the RAG system.
    
    FILENAME: Name of the file to remove
    """
    rag_system = ctx.obj['rag']
    
    # Check if document exists
    documents = rag_system.list_documents()
    doc_exists = any(doc.get('filename') == filename for doc in documents)
    
    if not doc_exists:
        click.echo(f"❌ Document '{filename}' not found in RAG system.")
        
        # Show available documents
        if documents:
            click.echo("Available documents:")
            for doc in documents:
                click.echo(f"   • {doc.get('filename', 'Unknown')}")
        
        sys.exit(1)
    
    # Confirmation
    if not confirm:
        if not click.confirm(f"Are you sure you want to remove '{filename}'?"):
            click.echo("Cancelled.")
            return
    
    click.echo(f"Removing document: {filename}")
    
    success = rag_system.remove_document(filename)
    
    if success:
        click.echo(f"✅ Successfully removed {filename}")
        
        # Show updated stats
        stats = rag_system.get_collection_stats()
        click.echo(f"📊 Collection now has {stats.get('total_documents', 0)} documents with {stats.get('total_chunks', 0)} chunks")
    else:
        click.echo("❌ Failed to remove document", err=True)
        sys.exit(1)


@main.command()
@click.option('--model', '-m', default='gemma3', help='Ollama model to use (default: gemma3)')
@click.option('--context-limit', default=5, help='Number of context chunks to retrieve (default: 5)')
@click.pass_context
def chat(ctx, model: str, context_limit: int):
    """Start an interactive chat session for generating professional writing.
    
    This command will:
    1. Prompt for company name or URL (with validation loop)
    2. Detect if input is URL (for webfetch) or company name (for websearch)
    3. Ask for your writing request
    4. Retrieve relevant context from your profile
    5. Generate professional content using Ollama
    6. Save the output to a file
    
    Note: Web search/fetch requires OLLAMA_API_KEY in .env file
    """
    rag_system = ctx.obj['rag']
    
    try:
        # Initialize Ollama client
        ollama_client = OllamaClient(model=model)
        
        # Initialize chat handler
        chat_handler = ChatHandler(rag_system, ollama_client)
        
        # Run interactive chat (no company/query passed - will prompt)
        chat_handler.run_interactive_session(
            company=None,
            query=None,
            context_limit=context_limit
        )
        
    except Exception as e:
        click.echo(f"❌ Error during chat session: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Number of results to return (default: 5)')
@click.option('--chunker', '-c', 
              type=click.Choice(['default', 'semantic', 'document_type']), 
              default='document_type',
              help='Chunking strategy for comparison')
@click.option('--multi-stage', '-ms', is_flag=True, help='Use multi-stage retrieval')
@click.option('--cluster', is_flag=True, help='Use cluster-based retrieval')
@click.option('--show-metadata', '-m', is_flag=True, help='Show chunk metadata')
@click.pass_context
def search(ctx, query: str, limit: int, chunker: str, multi_stage: bool, cluster: bool, show_metadata: bool):
    """Search your profile data for relevant information with enhanced retrieval.
    
    QUERY: Search query to find relevant context
    """
    # Initialize chunker
    chunker_map = {
        'default': DefaultChunker(),
        'semantic': SemanticChunker(),
        'document_type': DocumentTypeChunker()
    }
    
    rag_system = RAGSystem(chunker=chunker_map[chunker])
    
    click.echo(f"🔍 Searching for: '{query}'")
    click.echo(f"📋 Chunker: {chunker}")
    
    # Choose retrieval method
    if multi_stage:
        click.echo("🎯 Using multi-stage retrieval")
        results = rag_system.multi_stage_query(query, final_results=limit)
        if 'scores' in results:
            click.echo(f"📊 Relevance scores included")
    elif cluster:
        click.echo("🗂️  Using cluster-based retrieval")
        results = rag_system.cluster_based_retrieval(query, n_results=limit)
        if 'cluster_info' in results:
            click.echo(f"📊 Clusters: {results['cluster_info']}")
    else:
        results = rag_system.query(query, n_results=limit)
    
    click.echo()
    
    if not results['chunks']:
        click.echo("No relevant chunks found.")
        return
    
    click.echo(f"Found {len(results['chunks'])} relevant chunks:")
    click.echo()
    
    for i, (chunk, metadata) in enumerate(zip(results['chunks'], results['metadata'])):
        click.echo(f"📄 Result {i + 1}:")
        click.echo(f"   File: {metadata.get('filename', 'Unknown')}")
        click.echo(f"   Type: {metadata.get('document_type', 'unknown')}")
        click.echo(f"   Chunker: {metadata.get('chunker_type', 'unknown')}")
        click.echo(f"   Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('chunk_count', 1)}")
        
        if 'scores' in results:
            click.echo(f"   Score: {results['scores'][i]:.3f}")
        
        click.echo()
        click.echo(f"   Content: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
        
        if show_metadata:
            click.echo(f"   Metadata: {metadata}")
        
        click.echo()


@main.command()
@click.argument('path')
@click.option('--recursive', '-r', is_flag=True, help='Process subdirectories recursively')
@click.option('--patterns', '-p', multiple=True, help='File patterns to match (e.g., *.txt *.md)')
@click.option('--chunker', '-c', 
              type=click.Choice(['default', 'semantic', 'document_type']), 
              default='document_type',
              help='Chunking strategy to use')
@click.option('--metadata', '-m', multiple=True, help='Metadata key=value pairs')
@click.pass_context
def ingest_folder(ctx, path: str, recursive: bool, patterns: tuple, chunker: str, metadata: tuple):
    """Ingest all documents in a folder with enhanced processing.
    
    PATH: Path to the folder containing documents
    """
    # Parse metadata
    meta_dict = {}
    for item in metadata:
        if '=' in item:
            key, value = item.split('=', 1)
            meta_dict[key] = value
    
    # Initialize chunker
    chunker_map = {
        'default': DefaultChunker(),
        'semantic': SemanticChunker(),
        'document_type': DocumentTypeChunker()
    }
    
    rag = RAGSystem(chunker=chunker_map[chunker])
    
    # Convert patterns tuple to list
    file_patterns = list(patterns) if patterns else None
    
    click.echo(f"📁 Ingesting folder: {path}")
    click.echo(f"📋 Chunker: {chunker}")
    click.echo(f"🔄 Recursive: {recursive}")
    if file_patterns:
        click.echo(f"🎯 Patterns: {file_patterns}")
    
    results = rag.ingest_folder(path, recursive, file_patterns, meta_dict)
    
    # Display results
    click.echo(f"\n📊 Ingestion Results:")
    click.echo(f"   Total files found: {results['total_files']}")
    click.echo(f"   ✅ Successful: {results['summary']['total_successful']}")
    click.echo(f"   ❌ Failed: {results['summary']['total_failed']}")
    click.echo(f"   📈 Success rate: {results['summary']['success_rate']:.1%}")
    
    if results['failed']:
        click.echo(f"\n❌ Failed files:")
        for failed_file in results['failed']:
            click.echo(f"   - {failed_file}")


@main.command()
@click.argument('filename')
@click.pass_context
def verify(ctx, filename: str):
    """Verify source document integrity.
    
    FILENAME: Name of the file to verify
    """
    rag_system = ctx.obj['rag']
    integrity_check = rag_system.verify_source_integrity(filename)
    
    if integrity_check["status"] == "verified":
        click.echo(f"✅ {filename}: Source integrity preserved")
        click.echo(f"   📊 Chunks: {integrity_check['chunk_count']}")
        click.echo(f"   🔐 Hash: {integrity_check['original_hash'][:16]}...")
        click.echo(f"   ✓ Integrity: {integrity_check['integrity_preserved']}")
    elif integrity_check["status"] == "not_found":
        click.echo(f"❌ {filename}: Document not found in RAG system")
    else:
        click.echo(f"❌ {filename}: {integrity_check['status']}")


@main.command()
@click.pass_context
def stats(ctx):
    """Show comprehensive RAG system statistics."""
    rag_system = ctx.obj['rag']
    stats = rag_system.get_collection_stats()
    
    if not stats:
        click.echo("❌ Unable to retrieve statistics")
        return
    
    click.echo("📊 RAG System Statistics:")
    click.echo(f"   📄 Total documents: {stats.get('total_documents', 0)}")
    click.echo(f"   📝 Total chunks: {stats.get('total_chunks', 0)}")
    click.echo(f"   ✅ Original content chunks: {stats.get('original_content_chunks', 0)}")
    click.echo(f"   📏 Total characters: {stats.get('total_characters', 0):,}")
    click.echo(f"   🏷️  Collection name: {stats.get('collection_name', 'unknown')}")
    click.echo(f"   ⚙️  Chunker type: {stats.get('chunker_type', 'unknown')}")
    
    # Document types breakdown
    doc_types = stats.get('document_types', {})
    if doc_types:
        click.echo(f"\n📋 Document types:")
        for doc_type, count in doc_types.items():
            click.echo(f"   • {doc_type}: {count} chunks")


@main.command()
@click.argument('query')
@click.option('--max-results', '-n', default=5, help='Maximum number of results (default: 5, max: 10)')
@click.option('--api-key', help='Ollama API key (or set OLLAMA_API_KEY env var)')
@click.option('--json-output', '-j', is_flag=True, help='Output results as JSON')
def websearch(query: str, max_results: int, api_key: Optional[str], json_output: bool):
    """Search the web using Ollama's web search API.
    
    QUERY: Search query string
    
    Note: Requires OLLAMA_API_KEY environment variable or --api-key option.
    Get your API key at: https://ollama.com/settings/keys
    """
    try:
        web_search = OllamaWebSearch(api_key=api_key)
        
        click.echo(f"🔍 Searching the web for: '{query}'")
        click.echo()
        
        results = web_search.web_search(query, max_results)
        
        if json_output:
            import json
            output = {
                "query": query,
                "results": [r.to_dict() for r in results]
            }
            click.echo(json.dumps(output, indent=2))
        else:
            if not results:
                click.echo("No results found.")
                return
            
            click.echo(f"Found {len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                click.echo(f"📄 Result {i}: {result.title}")
                click.echo(f"   🔗 {result.url}")
                click.echo(f"   📝 {result.content[:200]}{'...' if len(result.content) > 200 else ''}")
                click.echo()
        
    except ValueError as e:
        click.echo(f"❌ {e}", err=True)
        click.echo("\nGet your API key at: https://ollama.com/settings/keys", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error during web search: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('url')
@click.option('--api-key', help='Ollama API key (or set OLLAMA_API_KEY env var)')
@click.option('--json-output', '-j', is_flag=True, help='Output result as JSON')
@click.option('--show-links', '-l', is_flag=True, help='Show all links found on the page')
def webfetch(url: str, api_key: Optional[str], json_output: bool, show_links: bool):
    """Fetch content from a URL using Ollama's web fetch API.
    
    URL: The URL to fetch content from
    
    Note: Requires OLLAMA_API_KEY environment variable or --api-key option.
    Get your API key at: https://ollama.com/settings/keys
    """
    try:
        web_search = OllamaWebSearch(api_key=api_key)
        
        click.echo(f"🌐 Fetching content from: {url}")
        click.echo()
        
        result = web_search.web_fetch(url)
        
        if json_output:
            import json
            output = result.to_dict()
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo(f"📄 Title: {result.title}")
            click.echo()
            click.echo(f"📝 Content:")
            click.echo(result.content)
            click.echo()
            
            if show_links:
                click.echo(f"🔗 Links found ({len(result.links)}):")
                for link in result.links:
                    click.echo(f"   • {link}")
            else:
                click.echo(f"🔗 {len(result.links)} links found (use --show-links to display)")
        
    except ValueError as e:
        click.echo(f"❌ {e}", err=True)
        click.echo("\nGet your API key at: https://ollama.com/settings/keys", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error during web fetch: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
