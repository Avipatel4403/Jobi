"""Utility classes for document processing and metadata extraction"""

import time
import hashlib
import mimetypes
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document reading and basic processing"""
    
    def __init__(self):
        self.supported_encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    
    def read_file(self, filepath: Path) -> Optional[str]:
        """Read file content with encoding detection and error handling"""
        try:
            # Try different encodings
            for encoding in self.supported_encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    if content.strip():
                        logger.debug(f"Successfully read {filepath} with {encoding} encoding")
                        return content
                        
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error reading {filepath} with {encoding}: {e}")
                    continue
            
            logger.error(f"Could not read {filepath} with any supported encoding")
            return None
            
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning while preserving original meaning"""
        # Remove excessive whitespace but preserve structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace but preserve intentional indentation
            if line.strip():
                cleaned_lines.append(line.rstrip())
            elif cleaned_lines and cleaned_lines[-1]:  # Preserve single empty lines
                cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines)


class MetadataExtractor:
    """Extracts comprehensive metadata from documents"""
    
    def extract_metadata(self, filepath: Path, content: str, 
                        user_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract comprehensive metadata from file and content
        
        Args:
            filepath: Path to the file
            content: File content
            user_metadata: User-provided metadata
            
        Returns:
            Complete metadata dictionary
        """
        # Basic file metadata
        stat = filepath.stat()
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        metadata = {
            # File information
            "filename": filepath.name,
            "filepath": str(filepath),
            "file_extension": filepath.suffix.lower(),
            "file_size": len(content),
            "file_size_bytes": stat.st_size,
            "source_file_hash": file_hash,
            
            # Timestamps
            "ingestion_time": time.time(),
            "file_modified_time": stat.st_mtime,
            "file_created_time": stat.st_ctime,
            
            # Content analysis
            "character_count": len(content),
            "word_count": len(content.split()),
            "line_count": len(content.split('\n')),
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
            
            # Content integrity
            "is_original_content": True,
            "content_modified": False,
            
            # Document type inference
            "document_type": self._infer_document_type(filepath, content),
            "mime_type": mimetypes.guess_type(str(filepath))[0] or "text/plain",
            
            # Content characteristics
            "has_code": self._detect_code_content(content),
            "has_structured_data": self._detect_structured_data(content),
            "language": self._detect_language(content),
            
            # Processing metadata
            "processor_version": "1.0",
            "extraction_time": time.time()
        }
        
        # Add user-provided metadata
        if user_metadata:
            metadata.update(user_metadata)
        
        return metadata
    
    def _infer_document_type(self, filepath: Path, content: str) -> str:
        """Infer document type from filename and content"""
        filename = filepath.name.lower()
        extension = filepath.suffix.lower()
        
        # Check filename patterns
        if any(keyword in filename for keyword in ['resume', 'cv']):
            return 'resume'
        elif any(keyword in filename for keyword in ['cover', 'letter']):
            return 'cover_letter'
        elif any(keyword in filename for keyword in ['project', 'portfolio']):
            return 'project'
        elif any(keyword in filename for keyword in ['profile', 'work_history', 'work history', 'summary']):
            return 'profile'
        elif 'readme' in filename:
            return 'documentation'
        
        # Check file extensions
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs'}
        if extension in code_extensions:
            return 'code'
        elif extension in {'.md', '.rst'}:
            return 'documentation'
        elif extension in {'.txt', '.doc', '.docx'}:
            return 'document'
        
        # Check content patterns
        if self._detect_code_content(content):
            return 'code'
        elif any(keyword in content.lower() for keyword in ['experience', 'education', 'skills', 'employment']):
            return 'resume'
        
        return 'document'
    
    def _detect_code_content(self, content: str) -> bool:
        """Detect if content contains code"""
        code_indicators = [
            'def ', 'function ', 'class ', 'import ', 'from ',
            '#include', 'public class', 'private ', 'protected ',
            '#!/', '<?php', '<script', 'SELECT ', 'INSERT ', 'UPDATE '
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in code_indicators)
    
    def _detect_structured_data(self, content: str) -> bool:
        """Detect if content contains structured data (JSON, XML, CSV, etc.)"""
        structured_indicators = [
            content.strip().startswith('{') and content.strip().endswith('}'),  # JSON
            content.strip().startswith('<') and content.strip().endswith('>'),  # XML
            ',' in content and '\n' in content and len(content.split(',')) > 5,  # CSV-like
            content.count('|') > 5 and content.count('\n') > 2,  # Table-like
        ]
        
        return any(structured_indicators)
    
    def _detect_language(self, content: str) -> str:
        """Simple language detection (can be enhanced with proper language detection libraries)"""
        # This is a very basic implementation
        # In practice, you might want to use libraries like langdetect or spacy
        
        english_indicators = ['the', 'and', 'for', 'are', 'with', 'have', 'this', 'that', 'from']
        
        words = content.lower().split()
        if len(words) < 10:
            return 'unknown'
        
        english_count = sum(1 for word in words[:100] if word in english_indicators)
        
        if english_count > 5:
            return 'english'
        
        return 'unknown'