"""Pluggable chunking strategies for different document types and use cases"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Split text into chunks
        
        Args:
            text: Text to chunk
            metadata: Optional metadata that might influence chunking
            
        Returns:
            List of text chunks
        """
        pass
    
    @abstractmethod
    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int, 
                          base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metadata for a specific chunk
        
        Args:
            chunk: The text chunk
            chunk_index: Index of this chunk
            total_chunks: Total number of chunks in document
            base_metadata: Base metadata from document
            
        Returns:
            Metadata dictionary for this chunk
        """
        pass


class DefaultChunker(BaseChunker):
    """Default chunking strategy with configurable size and overlap"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Split text into overlapping chunks at word boundaries"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int, 
                          base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic chunk metadata"""
        return {
            **base_metadata,
            "chunk_index": chunk_index,
            "chunk_count": total_chunks,
            "chunk_size": len(chunk),
            "chunker_type": "default"
        }


class SemanticChunker(BaseChunker):
    """Semantic chunking based on paragraphs and sections"""
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Split text by semantic boundaries (paragraphs, sections)"""
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size, finalize current chunk
            if (current_chunk and 
                len(current_chunk) + len(paragraph) + 2 > self.max_chunk_size):
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Current chunk too small, add paragraph anyway
                    current_chunk += "\n\n" + paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Handle cases where paragraphs are too long
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size:
                # Fall back to default chunking for oversized chunks
                default_chunker = DefaultChunker(self.max_chunk_size, 50)
                final_chunks.extend(default_chunker.chunk_text(chunk))
            else:
                final_chunks.append(chunk)
        
        return final_chunks or [text]  # Fallback to original text if no chunks
    
    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int, 
                          base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate semantic chunk metadata"""
        # Count paragraphs in chunk
        paragraph_count = len([p for p in chunk.split('\n\n') if p.strip()])
        
        return {
            **base_metadata,
            "chunk_index": chunk_index,
            "chunk_count": total_chunks,
            "chunk_size": len(chunk),
            "paragraph_count": paragraph_count,
            "chunker_type": "semantic"
        }


class CodeAwareChunker(BaseChunker):
    """Specialized chunker for code files that respects function/class boundaries"""
    
    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Split code by functions/classes while respecting syntax"""
        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # Check if this is a function/class definition
            is_definition = bool(re.match(r'^\s*(def |class |function |var |let |const )', line))
            
            # If adding this line would exceed max size and we're at a definition boundary
            if (current_size + line_size > self.max_chunk_size and 
                current_chunk_lines and is_definition):
                
                chunks.append('\n'.join(current_chunk_lines))
                current_chunk_lines = [line]
                current_size = line_size
            else:
                current_chunk_lines.append(line)
                current_size += line_size
        
        # Add final chunk
        if current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))
        
        return chunks or [text]
    
    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int, 
                          base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code-aware chunk metadata"""
        # Count functions/classes in chunk
        function_count = len(re.findall(r'^\s*def ', chunk, re.MULTILINE))
        class_count = len(re.findall(r'^\s*class ', chunk, re.MULTILINE))
        
        return {
            **base_metadata,
            "chunk_index": chunk_index,
            "chunk_count": total_chunks,
            "chunk_size": len(chunk),
            "function_count": function_count,
            "class_count": class_count,
            "chunker_type": "code_aware"
        }


class DocumentTypeChunker(BaseChunker):
    """Chunking strategy that adapts based on document type"""
    
    def __init__(self):
        self.chunkers = {
            'resume': SemanticChunker(min_chunk_size=200, max_chunk_size=800),
            'cover_letter': SemanticChunker(min_chunk_size=150, max_chunk_size=600),
            'project': DefaultChunker(chunk_size=600, overlap=75),
            'code': CodeAwareChunker(),
            'default': DefaultChunker()
        }
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Choose chunking strategy based on document type"""
        doc_type = 'default'
        
        if metadata:
            # Try to determine document type from metadata
            doc_type = metadata.get('document_type', 'default').lower()
            
            # Try to infer from filename
            if doc_type == 'default':
                filename = metadata.get('filename', '').lower()
                if 'resume' in filename or 'cv' in filename:
                    doc_type = 'resume'
                elif 'cover' in filename or 'letter' in filename:
                    doc_type = 'cover_letter'
                elif any(ext in filename for ext in ['.py', '.js', '.java', '.cpp']):
                    doc_type = 'code'
                elif 'project' in filename:
                    doc_type = 'project'
        
        chunker = self.chunkers.get(doc_type, self.chunkers['default'])
        return chunker.chunk_text(text, metadata)
    
    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int, 
                          base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate document-type-aware chunk metadata"""
        doc_type = base_metadata.get('document_type', 'default').lower()
        chunker = self.chunkers.get(doc_type, self.chunkers['default'])
        
        chunk_metadata = chunker.get_chunk_metadata(chunk, chunk_index, total_chunks, base_metadata)
        chunk_metadata['adaptive_chunker_type'] = doc_type
        
        return chunk_metadata


class CustomChunker(BaseChunker):
    """Template for users to create their own chunking strategies"""
    
    def __init__(self, chunk_function=None, metadata_function=None):
        """
        Initialize with custom functions
        
        Args:
            chunk_function: Function that takes (text, metadata) and returns List[str]
            metadata_function: Function that takes (chunk, index, total, base_metadata) and returns Dict
        """
        self.chunk_function = chunk_function or self._default_chunk
        self.metadata_function = metadata_function or self._default_metadata
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Use custom chunking function"""
        return self.chunk_function(text, metadata)
    
    def get_chunk_metadata(self, chunk: str, chunk_index: int, total_chunks: int, 
                          base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Use custom metadata function"""
        return self.metadata_function(chunk, chunk_index, total_chunks, base_metadata)
    
    def _default_chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Default fallback chunking"""
        return DefaultChunker().chunk_text(text, metadata)
    
    def _default_metadata(self, chunk: str, chunk_index: int, total_chunks: int, 
                         base_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Default fallback metadata"""
        return {
            **base_metadata,
            "chunk_index": chunk_index,
            "chunk_count": total_chunks,
            "chunker_type": "custom"
        }