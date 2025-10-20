"""
Ollama Web Search API Integration

This module provides access to Ollama's web search and web fetch capabilities.
"""

from .client import OllamaWebSearch, web_search, web_fetch
from .models import WebSearchResult, WebFetchResult

__all__ = [
    'OllamaWebSearch',
    'web_search',
    'web_fetch',
    'WebSearchResult',
    'WebFetchResult',
]
