"""
Ollama Web Search API client
"""

import os
import requests
from typing import Optional, List
import logging

from .models import WebSearchResult, WebFetchResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaWebSearch:
    """Client for Ollama's web search and web fetch APIs"""
    
    BASE_URL = "https://ollama.com/api"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Ollama Web Search client
        
        Args:
            api_key: Ollama API key. If not provided, uses OLLAMA_API_KEY env var
        
        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Ollama API key is required. Either pass it to the constructor or "
                "set the OLLAMA_API_KEY environment variable. "
                "Get your API key at: https://ollama.com/settings/keys"
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def web_search(
        self, 
        query: str, 
        max_results: int = 5
    ) -> List[WebSearchResult]:
        """Perform a web search using Ollama's API
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return (default 5, max 10)
        
        Returns:
            List of WebSearchResult objects
        
        Raises:
            ValueError: If API key is invalid
            requests.exceptions.RequestException: If the API request fails
        """
        if max_results > 10:
            logger.warning(f"max_results {max_results} exceeds maximum of 10, using 10")
            max_results = 10
        
        url = f"{self.BASE_URL}/web_search"
        payload = {
            "query": query,
            "max_results": max_results
        }
        
        try:
            logger.info(f"Searching for: '{query}' (max_results={max_results})")
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for item in data.get("results", []):
                results.append(WebSearchResult(
                    title=item["title"],
                    url=item["url"],
                    content=item["content"]
                ))
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during web search: {e}")
            if e.response.status_code == 401:
                raise ValueError(
                    "Invalid API key. Check your OLLAMA_API_KEY or get a new one at "
                    "https://ollama.com/settings/keys"
                )
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during web search: {e}")
            raise
    
    def web_fetch(self, url: str) -> WebFetchResult:
        """Fetch content from a specific URL using Ollama's API
        
        Args:
            url: The URL to fetch content from
        
        Returns:
            WebFetchResult object containing the page title, content, and links
        
        Raises:
            ValueError: If API key is invalid
            requests.exceptions.RequestException: If the API request fails
        """
        api_url = f"{self.BASE_URL}/web_fetch"
        payload = {"url": url}
        
        try:
            logger.info(f"Fetching content from: {url}")
            response = requests.post(api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            result = WebFetchResult(
                title=data["title"],
                content=data["content"],
                links=data["links"]
            )
            
            logger.info(f"Fetched '{result.title}' with {len(result.links)} links")
            return result
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during web fetch: {e}")
            if e.response.status_code == 401:
                raise ValueError(
                    "Invalid API key. Check your OLLAMA_API_KEY or get a new one at "
                    "https://ollama.com/settings/keys"
                )
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during web fetch: {e}")
            raise
    
    def search_and_summarize(
        self,
        query: str,
        max_results: int = 3
    ) -> str:
        """Search and return a formatted summary of results
        
        Args:
            query: The search query string
            max_results: Maximum number of results to return
        
        Returns:
            Formatted string with search results
        """
        results = self.web_search(query, max_results)
        
        if not results:
            return f"No results found for: {query}"
        
        summary = f"Search results for: {query}\n"
        summary += "=" * 60 + "\n\n"
        
        for i, result in enumerate(results, 1):
            summary += f"{i}. {result.title}\n"
            summary += f"   URL: {result.url}\n"
            summary += f"   {result.content[:200]}...\n\n"
        
        return summary


# Convenience functions for direct usage
_default_client: Optional[OllamaWebSearch] = None


def _get_default_client() -> OllamaWebSearch:
    """Get or create the default client"""
    global _default_client
    if _default_client is None:
        _default_client = OllamaWebSearch()
    return _default_client


def web_search(query: str, max_results: int = 5) -> List[WebSearchResult]:
    """Convenience function for web search
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default 5, max 10)
    
    Returns:
        List of WebSearchResult objects
        
    Example:
        >>> from jobi.web import web_search
        >>> results = web_search("what is ollama?", max_results=3)
        >>> for r in results:
        ...     print(f"{r.title}: {r.url}")
    """
    client = _get_default_client()
    return client.web_search(query, max_results)


def web_fetch(url: str) -> WebFetchResult:
    """Convenience function for web fetch
    
    Args:
        url: The URL to fetch content from
    
    Returns:
        WebFetchResult object
        
    Example:
        >>> from jobi.web import web_fetch
        >>> page = web_fetch("https://ollama.com")
        >>> print(page.title, len(page.links))
    """
    client = _get_default_client()
    return client.web_fetch(url)
