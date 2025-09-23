"""ChromaDB RAG System for managing user profile data and retrieval"""

import os
import chromadb
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG system using ChromaDB for local vector storage and retrieval"""
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """Initialize ChromaDB client and collection
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        self.collection_name = "user_profile"
        
        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}'")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "User profile data for RAG"}
            )
            logger.info(f"Created new collection '{self.collection_name}'")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            start = end - overlap
            
            if start >= len(text):
                break
                
        return chunks
    
    def ingest_document(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Ingest a document into ChromaDB
        
        Args:
            filepath: Path to the document to ingest
            metadata: Optional metadata to associate with the document
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"File {filepath} is empty")
                return False
            
            filename = Path(filepath).name
            
            # Prepare metadata
            doc_metadata = {
                "filename": filename,
                "filepath": filepath,
                "file_size": len(content),
                **(metadata or {})
            }
            
            # Chunk the document
            chunks = self._chunk_text(content)
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(filename, i, chunk)
                chunk_metadata = {
                    **doc_metadata,
                    "chunk_index": i,
                    "chunk_count": len(chunks)
                }
                
                documents.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {filename} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting document {filepath}: {e}")
            return False
    
    def _generate_chunk_id(self, filename: str, chunk_index: int, content: str) -> str:
        """Generate unique ID for a chunk"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{filename}_chunk_{chunk_index}_{content_hash}"
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store for relevant chunks
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            
        Returns:
            Dictionary containing query results
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Format results
            formatted_results = {
                "query": query_text,
                "chunks": [],
                "metadata": []
            }
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results["chunks"].append(doc)
                    formatted_results["metadata"].append(results['metadatas'][0][i])
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return {"query": query_text, "chunks": [], "metadata": []}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection
        
        Returns:
            List of document information
        """
        try:
            # Get all documents
            all_docs = self.collection.get()
            
            # Group by filename
            docs_by_file = {}
            
            for i, metadata in enumerate(all_docs['metadatas']):
                filename = metadata.get('filename', 'unknown')
                
                if filename not in docs_by_file:
                    docs_by_file[filename] = {
                        'filename': filename,
                        'filepath': metadata.get('filepath', 'unknown'),
                        'chunk_count': 0,
                        'total_size': 0
                    }
                
                docs_by_file[filename]['chunk_count'] += 1
                docs_by_file[filename]['total_size'] += len(all_docs['documents'][i])
            
            return list(docs_by_file.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def remove_document(self, filename: str) -> bool:
        """Remove a document from the collection
        
        Args:
            filename: Name of the file to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all documents with this filename
            all_docs = self.collection.get()
            
            ids_to_remove = []
            
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata.get('filename') == filename:
                    ids_to_remove.append(all_docs['ids'][i])
            
            if not ids_to_remove:
                logger.warning(f"No document found with filename: {filename}")
                return False
            
            # Remove documents
            self.collection.delete(ids=ids_to_remove)
            
            logger.info(f"Successfully removed {len(ids_to_remove)} chunks for {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document {filename}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            all_docs = self.collection.get()
            
            unique_files = set()
            total_chunks = len(all_docs['ids'])
            total_size = 0
            
            for i, metadata in enumerate(all_docs['metadatas']):
                unique_files.add(metadata.get('filename', 'unknown'))
                total_size += len(all_docs['documents'][i])
            
            return {
                "total_documents": len(unique_files),
                "total_chunks": total_chunks,
                "total_characters": total_size,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}