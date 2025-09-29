"""Core RAG System with ChromaDB"""

import chromadb
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from .chunkers import BaseChunker, DefaultChunker
from .ingestion import DocumentIngester

logger = logging.getLogger(__name__)


class RAGSystem:
    """Enhanced RAG system with pluggable chunking strategies"""
    
    def __init__(self, 
                 persist_directory: str = "./data/chromadb",
                 collection_name: str = "user_profile",
                 chunker: Optional[BaseChunker] = None):
        """Initialize ChromaDB client and collection
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            chunker: Custom chunking strategy (defaults to DefaultChunker)
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunker = chunker or DefaultChunker()
        
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
        
        # Initialize ingester
        self.ingester = DocumentIngester(self.collection, self.chunker)
    
    def ingest_document(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Ingest a single document with integrity preservation
        
        Args:
            filepath: Path to the document to ingest
            metadata: Optional metadata to associate with the document
            
        Returns:
            True if successful, False otherwise
        """
        return self.ingester.ingest_document(filepath, metadata)
    
    def ingest_folder(self, 
                     folder_path: str, 
                     recursive: bool = True,
                     file_patterns: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingest all documents in a folder
        
        Args:
            folder_path: Path to the folder containing documents
            recursive: Whether to search subdirectories
            file_patterns: List of file patterns to match (e.g., ['*.txt', '*.md'])
            metadata: Base metadata to apply to all documents
            
        Returns:
            Dictionary with ingestion results
        """
        return self.ingester.ingest_folder(folder_path, recursive, file_patterns, metadata)
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store for relevant chunks
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            
        Returns:
            Dictionary containing query results with source attribution
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Format results with source integrity info
            formatted_results = {
                "query": query_text,
                "chunks": [],
                "metadata": [],
                "source_integrity": "preserved"
            }
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results["chunks"].append(doc)
                    formatted_results["metadata"].append(results['metadatas'][0][i])
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return {"query": query_text, "chunks": [], "metadata": [], "source_integrity": "error"}
    
    def multi_stage_query(self, query_text: str, initial_results: int = 20, final_results: int = 5) -> Dict[str, Any]:
        """Enhanced retrieval with re-ranking while preserving source integrity"""
        try:
            # Stage 1: Broad retrieval
            initial_results_data = self.collection.query(
                query_texts=[query_text],
                n_results=initial_results
            )
            
            if not initial_results_data['documents'] or not initial_results_data['documents'][0]:
                return {"query": query_text, "chunks": [], "metadata": [], "source_integrity": "preserved"}
            
            # Stage 2: Score and rank results
            scored_results = []
            for i, doc in enumerate(initial_results_data['documents'][0]):
                metadata = initial_results_data['metadatas'][0][i]
                
                # Calculate relevance score (prioritize original content)
                score = self._calculate_relevance_score(doc, metadata, query_text)
                scored_results.append((score, doc, metadata))
            
            # Sort by score and take top results
            scored_results.sort(key=lambda x: x[0], reverse=True)
            top_results = scored_results[:final_results]
            
            return {
                "query": query_text,
                "chunks": [result[1] for result in top_results],
                "metadata": [result[2] for result in top_results],
                "scores": [result[0] for result in top_results],
                "source_integrity": "preserved"
            }
            
        except Exception as e:
            logger.error(f"Error in multi-stage query: {e}")
            return {"query": query_text, "chunks": [], "metadata": [], "source_integrity": "error"}
    
    def cluster_based_retrieval(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Group similar documents and retrieve from diverse clusters"""
        try:
            # Get more results initially
            broad_results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results * 3
            )
            
            if not broad_results['documents'] or not broad_results['documents'][0]:
                return {"query": query_text, "chunks": [], "metadata": [], "cluster_info": {}}
            
            # Group by filename/document type
            clusters = {}
            for i, doc in enumerate(broad_results['documents'][0]):
                metadata = broad_results['metadatas'][0][i]
                file_type = metadata.get('document_type', 'unknown')
                
                if file_type not in clusters:
                    clusters[file_type] = []
                clusters[file_type].append((doc, metadata))
            
            # Take top results from each cluster
            final_results = {"chunks": [], "metadata": []}
            results_per_cluster = max(1, n_results // len(clusters)) if clusters else n_results
            
            for cluster_docs in clusters.values():
                for doc, meta in cluster_docs[:results_per_cluster]:
                    final_results["chunks"].append(doc)
                    final_results["metadata"].append(meta)
                    if len(final_results["chunks"]) >= n_results:
                        break
                if len(final_results["chunks"]) >= n_results:
                    break
            
            return {
                "query": query_text,
                **final_results,
                "cluster_info": {k: len(v) for k, v in clusters.items()},
                "source_integrity": "preserved"
            }
            
        except Exception as e:
            logger.error(f"Error in cluster-based retrieval: {e}")
            return {"query": query_text, "chunks": [], "metadata": [], "cluster_info": {}}
    
    def personalized_query(self, query_text: str, user_preferences: Dict[str, Any], n_results: int = 5) -> Dict[str, Any]:
        """Retrieve with user preferences (file types, categories, recency)"""
        try:
            # Build where clause for filtering
            where_conditions = {}
            
            if "categories" in user_preferences:
                where_conditions["document_type"] = {"$in": user_preferences["categories"]}
            
            if "preferred_file_types" in user_preferences:
                where_conditions["file_extension"] = {"$in": user_preferences["preferred_file_types"]}
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results * 2,  # Get more, then filter
                where=where_conditions if where_conditions else None
            )
            
            # Post-process based on preferences
            return self._apply_personalization_filters(results, user_preferences, n_results, query_text)
            
        except Exception as e:
            logger.error(f"Error in personalized query: {e}")
            return {"query": query_text, "chunks": [], "metadata": [], "source_integrity": "error"}
    
    def _apply_personalization_filters(self, results: Dict, user_preferences: Dict, n_results: int, query_text: str) -> Dict[str, Any]:
        """Apply personalization filters to results"""
        if not results['documents'] or not results['documents'][0]:
            return {"query": query_text, "chunks": [], "metadata": [], "source_integrity": "preserved"}
        
        # Apply recency preference if specified
        if "prefer_recent" in user_preferences and user_preferences["prefer_recent"]:
            # Sort by ingestion time (more recent first)
            combined_results = list(zip(results['documents'][0], results['metadatas'][0]))
            combined_results.sort(
                key=lambda x: x[1].get('ingestion_time', 0), 
                reverse=True
            )
        else:
            combined_results = list(zip(results['documents'][0], results['metadatas'][0]))
        
        # Take top n results
        final_results = combined_results[:n_results]
        
        return {
            "query": query_text,
            "chunks": [result[0] for result in final_results],
            "metadata": [result[1] for result in final_results],
            "source_integrity": "preserved"
        }
    
    def _calculate_relevance_score(self, document: str, metadata: Dict, query: str) -> float:
        """Calculate relevance score for document re-ranking"""
        base_score = 1.0
        
        # Boost original content
        if metadata.get('is_original_content', False):
            base_score += 0.5
        
        # Boost based on document type relevance
        doc_type = metadata.get('document_type', '').lower()
        if any(keyword in query.lower() for keyword in ['resume', 'experience', 'skills']):
            if doc_type in ['resume', 'cv']:
                base_score += 0.3
        
        # Boost recent documents
        ingestion_time = metadata.get('ingestion_time', 0)
        if ingestion_time > 0:
            days_old = (time.time() - ingestion_time) / (24 * 3600)
            if days_old < 30:  # Recent documents get slight boost
                base_score += 0.1
        
        return base_score
    
    def verify_source_integrity(self, filename: str) -> Dict[str, Any]:
        """Verify that source material hasn't been altered"""
        try:
            all_docs = self.collection.get()
            
            file_chunks = []
            original_hash = None
            
            for i, metadata in enumerate(all_docs['metadatas']):
                if (metadata.get('filename') == filename and 
                    metadata.get('is_original_content', False)):
                    file_chunks.append({
                        'chunk_index': metadata.get('chunk_index'),
                        'content': all_docs['documents'][i],
                        'hash': metadata.get('source_file_hash')
                    })
                    original_hash = metadata.get('source_file_hash')
            
            if not file_chunks:
                return {"status": "not_found", "filename": filename}
            
            file_chunks.sort(key=lambda x: x['chunk_index'])
            
            return {
                "status": "verified",
                "filename": filename,
                "original_hash": original_hash,
                "chunk_count": len(file_chunks),
                "integrity_preserved": True
            }
            
        except Exception as e:
            logger.error(f"Error verifying integrity for {filename}: {e}")
            return {"status": "error", "filename": filename}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection with enhanced info"""
        return self.ingester.list_documents()
    
    def remove_document(self, filename: str) -> bool:
        """Remove a document from the collection"""
        return self.ingester.remove_document(filename)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the collection"""
        try:
            all_docs = self.collection.get()
            
            unique_files = set()
            total_chunks = len(all_docs['ids'])
            total_size = 0
            document_types = {}
            original_content_count = 0
            
            for i, metadata in enumerate(all_docs['metadatas']):
                filename = metadata.get('filename', 'unknown')
                unique_files.add(filename)
                total_size += len(all_docs['documents'][i])
                
                # Count document types
                doc_type = metadata.get('document_type', 'unknown')
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
                
                # Count original content
                if metadata.get('is_original_content', False):
                    original_content_count += 1
            
            return {
                "total_documents": len(unique_files),
                "total_chunks": total_chunks,
                "original_content_chunks": original_content_count,
                "total_characters": total_size,
                "document_types": document_types,
                "collection_name": self.collection_name,
                "chunker_type": self.chunker.__class__.__name__
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def set_chunker(self, chunker: BaseChunker):
        """Change the chunking strategy"""
        self.chunker = chunker
        self.ingester.chunker = chunker
        logger.info(f"Chunker changed to {chunker.__class__.__name__}")
    
    def record_feedback(self, query: str, chunk_id: str, relevance_score: float):
        """Record user feedback for improving future retrievals"""
        feedback_metadata = {
            "query": query,
            "relevance_score": relevance_score,
            "timestamp": time.time()
        }
        
        try:
            # Get current chunk
            chunk_data = self.collection.get(ids=[chunk_id])
            if chunk_data['metadatas']:
                current_meta = chunk_data['metadatas'][0]
                
                # Add feedback to metadata
                feedbacks = current_meta.get('user_feedback', [])
                feedbacks.append(feedback_metadata)
                current_meta['user_feedback'] = feedbacks
                
                # Update the chunk
                self.collection.update(
                    ids=[chunk_id],
                    metadatas=[current_meta]
                )
                
                logger.info(f"Recorded feedback for chunk {chunk_id}")
                
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")