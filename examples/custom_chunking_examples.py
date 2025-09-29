"""Example of creating custom chunking strategies for the enhanced RAG system"""

from jobi.rag import RAGSystem, CustomChunker
import re


def resume_specific_chunker(text: str, metadata=None):
    """Custom chunker specifically designed for resumes
    
    This chunker identifies resume sections (Experience, Education, Skills, etc.)
    and creates chunks based on these semantic boundaries.
    """
    sections = []
    current_section = ""
    
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Detect section headers (all caps, or specific keywords)
        is_section_header = (
            (line.isupper() and len(line) > 3) or 
            any(keyword in line.lower() for keyword in 
                ['experience', 'education', 'skills', 'projects', 'summary', 'objective', 'awards', 'certifications'])
        )
        
        if is_section_header:
            # Save previous section
            if current_section.strip():
                sections.append(current_section.strip())
            current_section = line + '\n'
        else:
            current_section += line + '\n'
    
    # Add final section
    if current_section.strip():
        sections.append(current_section.strip())
    
    return sections or [text]


def resume_metadata_enhancer(chunk: str, chunk_index: int, total_chunks: int, base_metadata: dict):
    """Custom metadata for resume chunks"""
    # Detect section type from first line
    first_line = chunk.split('\n')[0].lower()
    
    section_type = 'other'
    if 'experience' in first_line or 'work' in first_line:
        section_type = 'experience'
    elif 'education' in first_line:
        section_type = 'education'
    elif 'skill' in first_line:
        section_type = 'skills'
    elif 'project' in first_line:
        section_type = 'projects'
    elif 'summary' in first_line or 'objective' in first_line:
        section_type = 'summary'
    elif 'award' in first_line or 'honor' in first_line:
        section_type = 'awards'
    elif 'certification' in first_line or 'certificate' in first_line:
        section_type = 'certifications'
    
    # Extract skills if this is a skills section
    skills = []
    if section_type == 'skills':
        # Simple skill extraction (you can make this more sophisticated)
        skills = re.findall(r'\b[A-Z][a-z]*(?:\+\+|#|\.js|\.py)?\b', chunk)
        skills = [s for s in skills if len(s) > 2]  # Filter out short words
    
    return {
        **base_metadata,
        "chunk_index": chunk_index,
        "chunk_count": total_chunks,
        "section_type": section_type,
        "extracted_skills": skills,
        "chunker_type": "custom_resume"
    }


def code_project_chunker(text: str, metadata=None):
    """Custom chunker for code project descriptions
    
    Splits projects by clear boundaries like project names, descriptions, tech stacks
    """
    # Split by common project delimiters
    project_patterns = [
        r'\n\s*#+\s+',  # Markdown headers
        r'\n\s*\*\*[^*]+\*\*',  # Bold project names
        r'\n\s*Project\s*\d*:?',  # "Project 1:", "Project:", etc.
        r'\n\s*\d+\.\s+',  # Numbered lists
    ]
    
    # Try to split by patterns
    chunks = [text]
    for pattern in project_patterns:
        new_chunks = []
        for chunk in chunks:
            parts = re.split(pattern, chunk)
            if len(parts) > 1:
                new_chunks.extend([p.strip() for p in parts if p.strip()])
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
    
    # Ensure chunks aren't too long (max 800 chars)
    final_chunks = []
    for chunk in chunks:
        if len(chunk) > 800:
            # Split long chunks by paragraphs
            paragraphs = chunk.split('\n\n')
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > 800:
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += '\n\n' + para if current_chunk else para
            if current_chunk:
                final_chunks.append(current_chunk.strip())
        else:
            final_chunks.append(chunk)
    
    return final_chunks or [text]


def code_project_metadata_enhancer(chunk: str, chunk_index: int, total_chunks: int, base_metadata: dict):
    """Enhanced metadata for code project chunks"""
    # Extract technologies mentioned
    tech_keywords = [
        'python', 'javascript', 'react', 'node', 'django', 'flask', 'tensorflow',
        'pytorch', 'sql', 'mongodb', 'redis', 'docker', 'kubernetes', 'aws',
        'git', 'github', 'api', 'rest', 'graphql', 'microservices', 'machine learning',
        'deep learning', 'ai', 'nlp', 'computer vision', 'data science'
    ]
    
    chunk_lower = chunk.lower()
    found_technologies = [tech for tech in tech_keywords if tech in chunk_lower]
    
    # Detect if this chunk contains project metrics (numbers, percentages, etc.)
    has_metrics = bool(re.search(r'\d+%|\d+x|\d+\s*(users|customers|requests|ms|seconds)', chunk))
    
    # Detect URLs/links
    urls = re.findall(r'https?://[^\s]+', chunk)
    
    return {
        **base_metadata,
        "chunk_index": chunk_index,
        "chunk_count": total_chunks,
        "technologies": found_technologies,
        "has_metrics": has_metrics,
        "urls": urls,
        "chunker_type": "custom_code_project"
    }


def main():
    """Example usage of custom chunkers"""
    
    print("🚀 Custom Chunker Examples for Enhanced RAG System")
    print("=" * 60)
    
    # Example 1: Resume-specific chunker
    print("\n1. Resume-Specific Chunker")
    print("-" * 30)
    
    resume_chunker = CustomChunker(
        chunk_function=resume_specific_chunker,
        metadata_function=resume_metadata_enhancer
    )
    
    rag_resume = RAGSystem(chunker=resume_chunker)
    
    # Example usage (you would replace with actual file path)
    # success = rag_resume.ingest_document("resume.txt")
    # print(f"Resume ingestion successful: {success}")
    
    print("✅ Resume chunker created - splits by sections like Experience, Education, Skills")
    print("   Metadata includes: section_type, extracted_skills")
    
    # Example 2: Code project chunker
    print("\n2. Code Project Chunker")
    print("-" * 30)
    
    project_chunker = CustomChunker(
        chunk_function=code_project_chunker,
        metadata_function=code_project_metadata_enhancer
    )
    
    rag_projects = RAGSystem(chunker=project_chunker)
    
    print("✅ Code project chunker created - splits by project boundaries")
    print("   Metadata includes: technologies, has_metrics, urls")
    
    # Example 3: Combined approach with folder ingestion
    print("\n3. Batch Processing with Custom Chunkers")
    print("-" * 40)
    
    # You can ingest entire folders with custom chunking
    # results = rag_resume.ingest_folder(
    #     "path/to/documents",
    #     recursive=True,
    #     file_patterns=['*.txt', '*.md'],
    #     metadata={'batch_type': 'resume_docs'}
    # )
    
    print("✅ Custom chunkers work with folder ingestion")
    print("   Use ingest_folder() with your custom chunker for batch processing")
    
    # Example 4: Query with section-aware search
    print("\n4. Enhanced Querying")
    print("-" * 20)
    
    print("✅ Query results include custom metadata:")
    print("   • Section types (experience, skills, education)")
    print("   • Extracted technologies")
    print("   • Performance metrics detection")
    
    # Example query (would work after ingesting documents)
    # results = rag_resume.multi_stage_query("software engineering experience")
    # 
    # for chunk, metadata in zip(results["chunks"], results["metadata"]):
    #     print(f"Section: {metadata.get('section_type')}")
    #     print(f"Technologies: {metadata.get('technologies', [])}")
    #     print(f"Content: {chunk[:100]}...")
    #     print("---")
    
    print("\n🎯 Benefits of Custom Chunking:")
    print("   • Better semantic boundaries")
    print("   • Richer metadata for filtering")
    print("   • Domain-specific optimizations")
    print("   • Improved retrieval accuracy")
    
    print("\n📚 Next Steps:")
    print("   1. Create your own chunking functions")
    print("   2. Test with your specific document types")
    print("   3. Use enhanced metadata for better search")
    print("   4. Combine with multi-stage retrieval for best results")


if __name__ == "__main__":
    main()