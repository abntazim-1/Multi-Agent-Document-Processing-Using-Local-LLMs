"""
Summarizer Agent
Generates concise summaries from ingested data.
"""

import os
from typing import Optional
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from logging_config import get_logger

logger = get_logger("summarizer_agent")

# System prompt for summarizer agent
SUMMARIZER_SYSTEM_PROMPT = """You are a Summarizer Agent specialized in creating concise, accurate summaries from documents.

Your responsibilities include:
1. Analyzing document content and identifying key information
2. Generating concise bullet-point summaries that capture essential points
3. Maintaining accuracy and avoiding information loss
4. Adapting summary length and detail to the input document size
5. Ensuring summaries are clear and well-structured

You prioritize clarity, brevity, and completeness in your summaries."""

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class SummarizerResult:
    """Result from summarizer agent."""
    summary: str  # Generated summary text
    num_bullets: int  # Number of summary points
    summary_length: int  # Length of summary in characters
    success: bool
    error_message: Optional[str] = None


def run_summarizer(
    text: str,
    model_name: str = "tinyllama:1.1b",
    temperature: float = 0.2,
    num_bullets: int = 3
) -> SummarizerResult:
    """
    Run the summarizer agent to generate a concise summary from text.
    
    Args:
        text: Text content to summarize
        model_name: Ollama model to use for summarization
        temperature: Temperature setting for the model
        num_bullets: Number of bullet points in the summary
        
    Returns:
        SummarizerResult with generated summary and metadata
    """
    logger.info("="*60)
    logger.info("SUMMARIZER AGENT")
    logger.info("="*60)
    logger.info(f"System Prompt: {SUMMARIZER_SYSTEM_PROMPT[:100]}...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Input text length: {len(text)} characters")
    
    # Validate input
    if not text or not text.strip():
        error_msg = "Empty text provided to summarizer agent"
        logger.error(error_msg)
        return SummarizerResult(
            summary="",
            num_bullets=0,
            summary_length=0,
            success=False,
            error_message=error_msg
        )
    
    try:
        # Create LLM client
        llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template(
            "You are a professional summarizer. Analyze the following text and create a concise summary "
            "with exactly {num_bullets} bullet points. Each bullet should capture a key point or insight.\n\n"
            "Text to summarize:\n{text}\n\n"
            "Summary (exactly {num_bullets} bullets):"
        )
        
        # Invoke chain
        logger.info(f"Generating summary with {num_bullets} bullet points...")
        chain = prompt | llm
        response = chain.invoke({
            "text": text,
            "num_bullets": num_bullets
        })
        
        summary = response.content.strip()
        
        # Count actual bullets in output
        actual_bullets = summary.count('\n') + 1 if summary else 0
        if summary.startswith('-') or summary.startswith('*') or summary.startswith('•'):
            # Count list items
            lines = [line.strip() for line in summary.split('\n') if line.strip()]
            actual_bullets = len([line for line in lines if line.startswith(('-', '*', '•', '1.', '2.', '3.'))])
            if actual_bullets == 0:
                actual_bullets = len(lines)
        
        logger.info(f"Summary generated successfully ({len(summary)} characters, {actual_bullets} points)")
        logger.debug(f"Summary preview: {summary[:200]}...")
        
        return SummarizerResult(
            summary=summary,
            num_bullets=actual_bullets,
            summary_length=len(summary),
            success=True
        )
        
    except Exception as e:
        logger.exception(f"Error in summarizer agent: {e}")
        return SummarizerResult(
            summary="",
            num_bullets=0,
            summary_length=0,
            success=False,
            error_message=str(e)
        )

