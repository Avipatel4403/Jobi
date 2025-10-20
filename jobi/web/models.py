"""
Data models for Ollama Web Search API
"""

from typing import Dict, List, Any


class WebSearchResult:
    """Represents a single web search result"""
    
    def __init__(self, title: str, url: str, content: str):
        """Initialize a web search result
        
        Args:
            title: Title of the web page
            url: URL of the web page
            content: Relevant content snippet from the page
        """
        self.title = title
        self.url = url
        self.content = content
    
    def __repr__(self) -> str:
        return f"WebSearchResult(title='{self.title}', url='{self.url}')"
    
    def __str__(self) -> str:
        return f"{self.title}\n{self.url}\n{self.content[:100]}..."
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content
        }


class WebFetchResult:
    """Represents a web fetch result"""
    
    def __init__(self, title: str, content: str, links: List[str]):
        """Initialize a web fetch result
        
        Args:
            title: Title of the web page
            content: Main content of the page
            links: List of all links found on the page
        """
        self.title = title
        self.content = content
        self.links = links
    
    def __repr__(self) -> str:
        return f"WebFetchResult(title='{self.title}', links={len(self.links)} links)"
    
    def __str__(self) -> str:
        return f"{self.title}\n{len(self.links)} links found\n{self.content[:200]}..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary
        
        Returns:
            Dictionary representation of the result
        """
        return {
            "title": self.title,
            "content": self.content,
            "links": self.links
        }
