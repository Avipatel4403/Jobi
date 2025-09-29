"""Document ingestion with quality-preserving techniques"""

import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .chunkers import BaseChunker
from .utils import DocumentProcessor, MetadataExtractor

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles document ingestion with quality preservation and batch processing"""
    
    def __init__(self, collection, chunker: BaseChunker):
        self.collection = collection
        self.chunker = chunker
        self.processor = DocumentProcessor()
        self.metadata_extractor = MetadataExtractor()
    
    def ingest_document(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Ingest a single document with integrity preservation and quality enhancement
        
        Args:
            filepath: Path to the document to ingest
            metadata: Optional metadata to associate with the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return False
            
            # Read and process content
            content = self.processor.read_file(filepath)
            if not content:
                logger.warning(f"File {filepath} is empty or unreadable")
                return False
            
            # Extract enhanced metadata
            doc_metadata = self.metadata_extractor.extract_metadata(filepath, content, metadata)
            
            # Check for duplicates before processing
            if self._is_duplicate(filepath.name, doc_metadata['source_file_hash']):
                logger.info(f"Document {filepath.name} already exists with same content")
                return True
            
            # Remove existing version if filename exists but content is different
            self._remove_existing_document(filepath.name)
            
            # Chunk the document using the configured chunker
            chunks = self.chunker.chunk_text(content, doc_metadata)
            
            if not chunks:
                logger.warning(f"No chunks generated for {filepath}")
                return False
            
            # Prepare data for ChromaDB with enhanced metadata
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(filepath.name, i, chunk)
                chunk_metadata = self.chunker.get_chunk_metadata(
                    chunk, i, len(chunks), doc_metadata
                )
                
                documents.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            # Batch insert for better performance
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {filepath.name} with {len(chunks)} chunks using {self.chunker.__class__.__name__}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting document {filepath}: {e}")
            return False
    
    def ingest_folder(self, 
                     folder_path: str, 
                     recursive: bool = True,
                     file_patterns: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingest all documents in a folder with batch processing
        
        Args:
            folder_path: Path to the folder containing documents
            recursive: Whether to search subdirectories
            file_patterns: List of file patterns to match (e.g., ['*.txt', '*.md'])
            metadata: Base metadata to apply to all documents
            
        Returns:
            Dictionary with ingestion results
        """
        folder_path = Path(folder_path)
        
        if not folder_path.exists() or not folder_path.is_dir():
            logger.error(f"Folder not found or not a directory: {folder_path}")
            return {"success": False, "error": "Folder not found"}
        
        # Default file patterns
        if file_patterns is None:
            file_patterns = ['*.txt', '*.md', '*.pdf', '*.docx', '*.py', '*.js', '*.java', '*.cpp', '*.c', '*.h']
        
        # Find all matching files
        files_to_process = []
        for pattern in file_patterns:
            if recursive:
                files_to_process.extend(folder_path.rglob(pattern))
            else:
                files_to_process.extend(folder_path.glob(pattern))
        
        # Remove duplicates and sort
        files_to_process = sorted(list(set(files_to_process)))
        
        logger.info(f"Found {len(files_to_process)} files to process in {folder_path}")
        
        # Process files with progress tracking
        results = {
            "total_files": len(files_to_process),
            "successful": [],
            "failed": [],
            "skipped": [],
            "summary": {}
        }
        
        for file_path in files_to_process:
            try:
                # Prepare file-specific metadata
                file_metadata = {
                    "batch_ingestion": True,
                    "source_folder": str(folder_path),
                    **(metadata or {})
                }
                
                # Add relative path info
                try:
                    relative_path = file_path.relative_to(folder_path)
                    file_metadata["relative_path"] = str(relative_path)
                    file_metadata["folder_depth"] = len(relative_path.parts) - 1
                except ValueError:
                    pass
                
                # Ingest the file
                if self.ingest_document(str(file_path), file_metadata):
                    results["successful"].append(str(file_path))
                else:
                    results["failed"].append(str(file_path))
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results["failed"].append(str(file_path))
        
        # Generate summary
        results["summary"] = {
            "success_rate": len(results["successful"]) / len(files_to_process) if files_to_process else 0,
            "total_successful": len(results["successful"]),
            "total_failed": len(results["failed"]),
            "total_skipped": len(results["skipped"])
        }
        
        logger.info(f"Folder ingestion complete: {results['summary']['total_successful']}/{results['total_files']} files successful")
        
        return results
    
    def _is_duplicate(self, filename: str, file_hash: str) -> bool:
        """Check if document with same content already exists"""
        try:
            all_docs = self.collection.get()
            
            for metadata in all_docs['metadatas']:
                if (metadata.get('filename') == filename and 
                    metadata.get('source_file_hash') == file_hash):
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
            return False
    
    def _remove_existing_document(self, filename: str):
        """Remove existing document with same filename"""
        try:
            all_docs = self.collection.get()
            
            ids_to_remove = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('filename') == filename:
                    ids_to_remove.append(all_docs['ids'][i])
            
            if ids_to_remove:
                self.collection.delete(ids=ids_to_remove)
                logger.info(f"Removed existing version of {filename}")
                
        except Exception as e:
            logger.error(f"Error removing existing document: {e}")
    
    def _generate_chunk_id(self, filename: str, chunk_index: int, content: str) -> str:
        """Generate unique ID for a chunk"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{filename}_chunk_{chunk_index}_{content_hash}"
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection with enhanced information"""
        try:
            all_docs = self.collection.get()
            
            docs_by_file = {}
            
            for i, metadata in enumerate(all_docs['metadatas']):
                filename = metadata.get('filename', 'unknown')
                
                if filename not in docs_by_file:
                    docs_by_file[filename] = {
                        'filename': filename,
                        'filepath': metadata.get('filepath', 'unknown'),
                        'document_type': metadata.get('document_type', 'unknown'),
                        'chunk_count': 0,
                        'total_size': 0,
                        'chunker_type': metadata.get('chunker_type', 'unknown'),
                        'ingestion_time': metadata.get('ingestion_time'),
                        'file_hash': metadata.get('source_file_hash', 'unknown')[:16] + '...' if metadata.get('source_file_hash') else 'unknown',
                        'is_original_content': metadata.get('is_original_content', True)
                    }
                
                docs_by_file[filename]['chunk_count'] += 1
                docs_by_file[filename]['total_size'] += len(all_docs['documents'][i])
            
            return list(docs_by_file.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def remove_document(self, filename: str) -> bool:
        """Remove a document from the collection"""
        try:
            all_docs = self.collection.get()
            
            ids_to_remove = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('filename') == filename:
                    ids_to_remove.append(all_docs['ids'][i])
            
            if not ids_to_remove:
                logger.warning(f"No document found with filename: {filename}")
                return False
            
            self.collection.delete(ids=ids_to_remove)
            logger.info(f"Successfully removed {len(ids_to_remove)} chunks for {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {filename}: {e}")
            return False