"""
Reporter Agent
Converts decisions into natural, readable explanations.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from logging_config import get_logger

logger = get_logger("reporter_agent")

# System prompt for reporter agent
REPORTER_SYSTEM_PROMPT = """You are a Reporter Agent specialized in converting technical decisions into natural, readable explanations.

Your responsibilities include:
1. Translating decision outputs into clear, professional language
2. Creating human-friendly explanations that stakeholders can understand
3. Maintaining accuracy while improving readability
4. Structuring information in a logical, easy-to-follow format
5. Ensuring tone is appropriate for the target audience

You excel at making complex information accessible and actionable for non-technical audiences."""

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class ReporterResult:
    """Result from reporter agent."""
    report: str  # Natural language report
    report_length: int  # Length of report in characters
    word_count: int  # Word count of the report
    success: bool
    error_message: Optional[str] = None


def run_reporter(
    summary: str,
    decision: str,
    model_name: str = "phi3.5:latest",
    temperature: float = 0.2,
    max_words: int = 150
) -> ReporterResult:
    """
    Run the reporter agent to convert decisions into natural explanations.
    
    Args:
        summary: Summary text from the summarizer agent
        decision: Decision text from the decision agent
        model_name: Ollama model to use for reporting
        temperature: Temperature setting for the model
        max_words: Maximum word count for the report
        
    Returns:
        ReporterResult with natural language report
    """
    logger.info("="*60)
    logger.info("REPORTER AGENT")
    logger.info("="*60)
    logger.info(f"System Prompt: {REPORTER_SYSTEM_PROMPT[:100]}...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Summary length: {len(summary)} characters")
    logger.info(f"Decision length: {len(decision)} characters")
    
    # Validate input
    if not summary or not summary.strip():
        error_msg = "Empty summary provided to reporter agent"
        logger.error(error_msg)
        return ReporterResult(
            report="",
            report_length=0,
            word_count=0,
            success=False,
            error_message=error_msg
        )
    
    if not decision or not decision.strip():
        error_msg = "Empty decision provided to reporter agent"
        logger.error(error_msg)
        return ReporterResult(
            report="",
            report_length=0,
            word_count=0,
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
            "You are a professional reporter. Convert the following summary and decision into a "
            "concise, natural-language report suitable for stakeholders. "
            "The report should be clear, professional, and easy to understand.\n\n"
            "Limit the report to approximately {max_words} words.\n\n"
            "Summary:\n{summary}\n\n"
            "Decision & Analysis:\n{decision}\n\n"
            "Professional Report:"
        )
        
        # Invoke chain
        logger.info(f"Generating natural language report (max {max_words} words)...")
        chain = prompt | llm
        response = chain.invoke({
            "summary": summary,
            "decision": decision,
            "max_words": max_words
        })
        
        report_text = response.content.strip()
        word_count = len(report_text.split())
        
        logger.info(f"Report generated successfully ({len(report_text)} characters, {word_count} words)")
        logger.debug(f"Report preview: {report_text[:200]}...")
        
        return ReporterResult(
            report=report_text,
            report_length=len(report_text),
            word_count=word_count,
            success=True
        )
        
    except Exception as e:
        logger.exception(f"Error in reporter agent: {e}")
        return ReporterResult(
            report="",
            report_length=0,
            word_count=0,
            success=False,
            error_message=str(e)
        )

