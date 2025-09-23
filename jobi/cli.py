"""CLI interface for Jobi - LLM CLI tool for professional writing"""

import click
import sys
from pathlib import Path
from typing import Optional

from .rag import RAGSystem
from .ollama_client import OllamaClient
from .chat import ChatHandler


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
    
    # Initialize RAG system
    try:
        ctx.obj['rag'] = RAGSystem()
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
@click.option('--company', '-c', help='Company name or URL')
@click.option('--query', '-q', help='Your query/request')
@click.option('--context-limit', default=5, help='Number of context chunks to retrieve (default: 5)')
@click.pass_context
def chat(ctx, model: str, company: Optional[str], query: Optional[str], context_limit: int):
    """Start an interactive chat session for generating professional writing.
    
    This command will:
    1. Ask for company information (if not provided)
    2. Ask for your writing request (if not provided)
    3. Retrieve relevant context from your profile
    4. Generate professional content using Ollama
    5. Save the output to a file
    """
    rag_system = ctx.obj['rag']
    
    try:
        # Initialize Ollama client
        ollama_client = OllamaClient(model=model)
        
        # Initialize chat handler
        chat_handler = ChatHandler(rag_system, ollama_client)
        
        # Run interactive chat
        chat_handler.run_interactive_session(
            company=company,
            query=query,
            context_limit=context_limit
        )
        
    except Exception as e:
        click.echo(f"❌ Error during chat session: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Number of results to return (default: 5)')
@click.option('--show-metadata', '-m', is_flag=True, help='Show chunk metadata')
@click.pass_context
def search(ctx, query: str, limit: int, show_metadata: bool):
    """Search your profile data for relevant information.
    
    QUERY: Search query to find relevant context
    """
    rag_system = ctx.obj['rag']
    
    click.echo(f"🔍 Searching for: '{query}'")
    click.echo()
    
    results = rag_system.query(query, n_results=limit)
    
    if not results['chunks']:
        click.echo("No relevant chunks found.")
        return
    
    click.echo(f"Found {len(results['chunks'])} relevant chunks:")
    click.echo()
    
    for i, (chunk, metadata) in enumerate(zip(results['chunks'], results['metadata'])):
        click.echo(f"📄 Result {i + 1}:")
        click.echo(f"   File: {metadata.get('filename', 'Unknown')}")
        click.echo(f"   Chunk: {metadata.get('chunk_index', 0) + 1}/{metadata.get('chunk_count', 1)}")
        click.echo()
        click.echo(f"   Content: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
        
        if show_metadata:
            click.echo(f"   Metadata: {metadata}")
        
        click.echo()


if __name__ == '__main__':
    main()