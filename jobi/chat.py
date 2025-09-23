"""Chat handler for interactive professional writing generation"""

import click
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
import requests
from bs4 import BeautifulSoup
import logging

from .rag import RAGSystem
from .ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatHandler:
    """Handles interactive chat sessions for generating professional writing"""
    
    def __init__(self, rag_system: RAGSystem, ollama_client: OllamaClient):
        """Initialize chat handler
        
        Args:
            rag_system: RAG system for context retrieval
            ollama_client: Ollama client for LLM generation
        """
        self.rag_system = rag_system
        self.ollama_client = ollama_client
        self.outputs_dir = Path("./outputs")
        self.outputs_dir.mkdir(exist_ok=True)
    
    def run_interactive_session(
        self, 
        company: Optional[str] = None, 
        query: Optional[str] = None,
        context_limit: int = 5
    ):
        """Run an interactive chat session
        
        Args:
            company: Company name or URL (if not provided, will prompt)
            query: User query (if not provided, will prompt)
            context_limit: Maximum number of context chunks to retrieve
        """
        click.echo("🚀 Welcome to Jobi - Professional Writing Assistant")
        click.echo()
        
        # Check if we have any documents in RAG system
        stats = self.rag_system.get_collection_stats()
        if stats.get('total_documents', 0) == 0:
            click.echo("⚠️  No documents found in your profile.")
            click.echo("   Use 'jobi ingest <file>' to add your resume, work history, etc.")
            click.echo()
        else:
            click.echo(f"📊 Your profile: {stats.get('total_documents', 0)} documents, {stats.get('total_chunks', 0)} chunks")
            click.echo()
        
        # Get company information
        if not company:
            company = click.prompt("🏢 Company name or URL", type=str)
        
        company_info = self._get_company_info(company)
        
        # Get user query
        if not query:
            query = click.prompt("✍️  What would you like me to write", type=str)
        
        # Retrieve relevant context
        click.echo("\n🔍 Retrieving relevant context from your profile...")
        
        context_results = self.rag_system.query(
            f"{query} {company_info.get('name', '')}",
            n_results=context_limit
        )
        
        if context_results['chunks']:
            click.echo(f"   Found {len(context_results['chunks'])} relevant chunks")
            
            # Show context sources
            sources = set()
            for metadata in context_results['metadata']:
                sources.add(metadata.get('filename', 'unknown'))
            
            click.echo(f"   Sources: {', '.join(sources)}")
        else:
            click.echo("   No relevant context found in your profile")
        
        click.echo()
        
        # Generate response
        click.echo("🤖 Generating professional content...")
        click.echo()
        
        full_response = self._generate_response(
            company_info=company_info,
            user_query=query,
            context_results=context_results
        )
        
        # Display response
        click.echo("📄 Generated Content:")
        click.echo("=" * 60)
        click.echo(full_response)
        click.echo("=" * 60)
        click.echo()
        
        # Save response
        filename = self._save_response(
            company_name=company_info.get('name', 'unknown'),
            query=query,
            content=full_response
        )
        
        click.echo(f"💾 Saved to: {filename}")
        click.echo()
        
        # Ask if user wants to regenerate or modify
        if click.confirm("Would you like to regenerate with different parameters?"):
            self.run_interactive_session(company=company, context_limit=context_limit)
    
    def _get_company_info(self, company_input: str) -> dict:
        """Get company information from name or URL
        
        Args:
            company_input: Company name or URL
            
        Returns:
            Dictionary with company information
        """
        company_info = {"name": company_input, "description": ""}
        
        # Check if input looks like a URL
        if company_input.startswith(('http://', 'https://', 'www.')):
            url = company_input if company_input.startswith('http') else f"https://{company_input}"
            
            try:
                click.echo(f"🌐 Fetching company info from: {url}")
                
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract basic info
                title = soup.find('title')
                if title:
                    company_info['name'] = title.get_text().strip()
                
                # Try to get description from meta tags
                description = soup.find('meta', attrs={'name': 'description'})
                if not description:
                    description = soup.find('meta', attrs={'property': 'og:description'})
                
                if description:
                    company_info['description'] = description.get('content', '').strip()
                
                # Get some text content
                paragraphs = soup.find_all('p')
                text_content = ' '.join([p.get_text().strip() for p in paragraphs[:3]])
                if text_content and len(text_content) > len(company_info['description']):
                    company_info['description'] = text_content[:500] + "..."
                
                company_info['url'] = url
                
                click.echo(f"   ✅ Found: {company_info['name']}")
                
            except Exception as e:
                logger.warning(f"Could not fetch company info from URL: {e}")
                click.echo(f"   ⚠️  Could not fetch info from URL: {e}")
                
                # Ask user to provide company info manually
                name = click.prompt("Company name", default=company_input)
                description = click.prompt("Company description (optional)", default="", show_default=False)
                
                company_info['name'] = name
                company_info['description'] = description
        
        return company_info
    
    def _generate_response(self, company_info: dict, user_query: str, context_results: dict) -> str:
        """Generate response using Ollama
        
        Args:
            company_info: Company information
            user_query: User's writing request
            context_results: Retrieved context from RAG system
            
        Returns:
            Generated response text
        """
        # Build system message
        system_message = """You are a professional writing assistant that helps create high-quality cover letters, cold emails, and application responses. 

Your task is to:
1. Use the provided personal context (resume, work history, skills) to highlight relevant qualifications
2. Tailor the content to the specific company and role
3. Write in a professional, confident, and personalized tone
4. Be specific and concrete rather than generic
5. Keep the content focused and concise
6. Ensure proper formatting and structure

Always write from the first person perspective and make the content feel authentic and personalized."""
        
        # Build user prompt
        prompt_parts = []
        
        # Add company information
        if company_info.get('name'):
            prompt_parts.append(f"Company: {company_info['name']}")
        
        if company_info.get('description'):
            prompt_parts.append(f"Company Description: {company_info['description']}")
        
        if company_info.get('url'):
            prompt_parts.append(f"Company Website: {company_info['url']}")
        
        # Add user request
        prompt_parts.append(f"\\nRequest: {user_query}")
        
        # Add personal context
        if context_results['chunks']:
            prompt_parts.append("\\nRelevant Personal Context:")
            for i, chunk in enumerate(context_results['chunks']):
                metadata = context_results['metadata'][i]
                source = metadata.get('filename', 'unknown')
                prompt_parts.append(f"\\nFrom {source}:")
                prompt_parts.append(chunk)
        
        prompt_parts.append("\\nPlease generate the requested professional content based on the above information.")
        
        full_prompt = "\\n".join(prompt_parts)
        
        # Generate response with streaming
        response_parts = []
        
        try:
            for chunk in self.ollama_client.generate_response(
                prompt=full_prompt,
                system_message=system_message,
                stream=True,
                temperature=0.7
            ):
                click.echo(chunk, nl=False)  # Print as it streams
                response_parts.append(chunk)
            
            click.echo()  # New line after streaming
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error generating response: {e}"
        
        return ''.join(response_parts)
    
    def _save_response(self, company_name: str, query: str, content: str) -> str:
        """Save generated response to file
        
        Args:
            company_name: Name of the company
            query: User query/request
            content: Generated content
            
        Returns:
            Filename of saved file
        """
        # Create safe filename
        safe_company = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name.lower())
        safe_query = re.sub(r'[^a-zA-Z0-9_-]', '_', query.lower())
        
        # Truncate if too long
        safe_company = safe_company[:30]
        safe_query = safe_query[:30]
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        filename = f"{safe_company}_{safe_query}_{timestamp}.txt"
        filepath = self.outputs_dir / filename
        
        # Save content
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Company: {company_name}\n")
                f.write(f"Request: {query}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                f.write(content)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving response: {e}")
            return f"Error saving to file: {e}"