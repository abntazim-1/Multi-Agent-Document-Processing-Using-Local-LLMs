"""
Document Ingestion Agent
Handles loading, cleaning, and chunking of documents.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from logging_config import get_logger

logger = get_logger("doc_ingestion_agent")

# Try to import text splitter - fallback to simple chunking if not available
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_TEXT_SPLITTER = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        HAS_TEXT_SPLITTER = True
    except ImportError:
        HAS_TEXT_SPLITTER = False
        logger.warning("Text splitter not available, using simple chunking")

# System prompt for document ingestion
DOC_INGESTION_SYSTEM_PROMPT = """You are a Document Ingestion Agent specialized in processing and preparing text documents for analysis.

Your responsibilities include:
1. Loading text files from specified directories
2. Cleaning and normalizing text content
3. Chunking documents into manageable segments for processing
4. Ensuring data quality and handling encoding issues
5. Preparing structured data for downstream agents

You maintain high standards for data quality and provide detailed metadata about processed documents."""

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class DocIngestionResult:
    """Result from document ingestion agent."""
    documents: List[Dict[str, str]]  # List of {filename, text, filepath}
    combined_text: str  # Combined text from all documents
    chunks: List[str]  # Chunked text segments
    metadata: Dict  # Metadata about ingestion process
    success: bool
    error_message: Optional[str] = None


def load_documents(data_dir: Path) -> List[Dict[str, str]]:
    """
    Load all text files from the specified directory.
    
    Args:
        data_dir: Path to directory containing text files
        
    Returns:
        List of dictionaries with filename, text, and filepath
    """
    documents = []
    data_path = Path(data_dir).resolve()
    
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_path}")
        return documents
    
    logger.info(f"Reading files from: {data_path}")
    
    # Get all .txt files
    txt_files = list(data_path.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} .txt files: {[f.name for f in txt_files]}")
    
    for file_path in txt_files:
        try:
            # Read with error handling and encoding fallback
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {file_path.name}, trying latin-1")
                text = file_path.read_text(encoding="latin-1")
            
            # Strip whitespace but keep structure
            text = text.strip()
            
            if not text:
                logger.warning(f"File {file_path.name} is empty, skipping")
                continue
            
            documents.append({
                "filename": file_path.name,
                "text": text,
                "filepath": str(file_path)
            })
            logger.info(f"Successfully read {file_path.name} ({len(text)} characters)")
            
        except Exception as e:
            logger.error(f"Error reading file {file_path.name}: {e}")
            continue
    
    return documents


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Normalize multiple spaces to single space
    import re
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    return cleaned_text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    if HAS_TEXT_SPLITTER:
        # Use LangChain's text splitter if available
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
    else:
        # Simple chunking fallback
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
            if start >= len(text):
                break
    
    logger.info(f"Split text into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    return chunks


def combine_documents(documents: List[Dict[str, str]]) -> str:
    """
    Combine all documents into a single text string with proper formatting.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Combined text string
    """
    if not documents:
        logger.warning("No documents to combine")
        return ""
    
    combined_parts = []
    for doc in documents:
        filename = doc.get('filename', 'unknown')
        text = doc.get('text', '')
        
        if not text or not text.strip():
            logger.warning(f"Skipping empty document: {filename}")
            continue
        
        # Format: filename header followed by content
        combined_parts.append(f"=== {filename} ===\n{text.strip()}\n")
    
    if not combined_parts:
        logger.error("No valid document content to combine")
        return ""
    
    combined_text = "\n".join(combined_parts).strip()
    logger.info(f"Combined {len(documents)} documents into {len(combined_text)} characters")
    
    return combined_text


def run_doc_ingestion(
    data_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    enable_cleaning: bool = True
) -> DocIngestionResult:
    """
    Run the document ingestion agent to load, clean, and chunk documents.
    
    Args:
        data_dir: Path to directory containing text files
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        enable_cleaning: Whether to clean and normalize text
        
    Returns:
        DocIngestionResult with processed documents and metadata
    """
    logger.info("="*60)
    logger.info("DOCUMENT INGESTION AGENT")
    logger.info("="*60)
    logger.info(f"System Prompt: {DOC_INGESTION_SYSTEM_PROMPT[:100]}...")
    
    try:
        # Load documents
        logger.info("Loading documents...")
        documents = load_documents(data_dir)
        
        if not documents:
            error_msg = f"No documents found in {data_dir}"
            logger.error(error_msg)
            return DocIngestionResult(
                documents=[],
                combined_text="",
                chunks=[],
                metadata={"error": error_msg},
                success=False,
                error_message=error_msg
            )
        
        # Clean documents if enabled
        if enable_cleaning:
            logger.info("Cleaning documents...")
            for doc in documents:
                doc['text'] = clean_text(doc['text'])
        
        # Combine documents
        logger.info("Combining documents...")
        combined_text = combine_documents(documents)
        
        if not combined_text:
            error_msg = "No valid content after combining documents"
            logger.error(error_msg)
            return DocIngestionResult(
                documents=documents,
                combined_text="",
                chunks=[],
                metadata={"error": error_msg},
                success=False,
                error_message=error_msg
            )
        
        # Chunk text
        logger.info("Chunking text...")
        chunks = chunk_text(combined_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Prepare metadata
        metadata = {
            "num_documents": len(documents),
            "total_characters": len(combined_text),
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "cleaning_enabled": enable_cleaning,
            "file_names": [doc['filename'] for doc in documents]
        }
        
        logger.info(f"Document ingestion completed successfully")
        logger.info(f"  Documents: {metadata['num_documents']}")
        logger.info(f"  Total characters: {metadata['total_characters']}")
        logger.info(f"  Chunks: {metadata['num_chunks']}")
        
        return DocIngestionResult(
            documents=documents,
            combined_text=combined_text,
            chunks=chunks,
            metadata=metadata,
            success=True
        )
        
    except Exception as e:
        logger.exception(f"Error in document ingestion agent: {e}")
        return DocIngestionResult(
            documents=[],
            combined_text="",
            chunks=[],
            metadata={},
            success=False,
            error_message=str(e)
        )

