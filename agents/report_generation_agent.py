"""
Report Generation Agent
Compiles all outputs into a final structured report.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from logging_config import get_logger

logger = get_logger("report_generation_agent")

# System prompt for report generation agent
REPORT_GENERATION_SYSTEM_PROMPT = """You are a Report Generation Agent specialized in compiling and structuring final reports.

Your responsibilities include:
1. Combining outputs from all previous agents into a cohesive report
2. Structuring the report with clear sections and formatting
3. Adding metadata, timestamps, and execution details
4. Ensuring the report is professional and complete
5. Saving the report in both human-readable and machine-readable formats

You create well-organized, comprehensive reports that document the entire workflow execution."""

@dataclass
class ReportGenerationResult:
    """Result from report generation agent."""
    report_path: str  # Path to the saved report file
    json_path: str  # Path to the JSON report file
    report_content: str  # Full report content
    metadata: Dict[str, Any]  # Report metadata
    success: bool
    error_message: Optional[str] = None


def create_structured_report(
    summary: str,
    decision: str,
    reporter_output: str,
    ingestion_metadata: Dict[str, Any],
    execution_timestamp: str
) -> str:
    """
    Create a structured report combining all agent outputs.
    
    Args:
        summary: Summary from summarizer agent
        decision: Decision from decision agent
        reporter_output: Report from reporter agent
        ingestion_metadata: Metadata from document ingestion
        execution_timestamp: Timestamp of workflow execution
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "="*80,
        "AGENTIC WORKFLOW EXECUTION REPORT",
        "="*80,
        f"Generated: {execution_timestamp}",
        "",
        "EXECUTIVE SUMMARY",
        "-"*80,
        reporter_output,
        "",
        "DETAILED ANALYSIS",
        "-"*80,
        "",
        "1. DOCUMENT SUMMARY",
        "-"*40,
        summary,
        "",
        "2. DECISION ANALYSIS",
        "-"*40,
        decision,
        "",
        "WORKFLOW METADATA",
        "-"*80,
        f"Documents Processed: {ingestion_metadata.get('num_documents', 'N/A')}",
        f"Total Characters: {ingestion_metadata.get('total_characters', 'N/A')}",
        f"Chunks Created: {ingestion_metadata.get('num_chunks', 'N/A')}",
        f"Processing Date: {execution_timestamp}",
        "",
        "="*80,
        "End of Report",
        "="*80
    ]
    
    return "\n".join(report_lines)


def create_json_report(
    summary: str,
    decision: str,
    reporter_output: str,
    ingestion_metadata: Dict[str, Any],
    execution_timestamp: str
) -> Dict[str, Any]:
    """
    Create a JSON-structured report for machine readability.
    
    Args:
        summary: Summary from summarizer agent
        decision: Decision from decision agent
        reporter_output: Report from reporter agent
        ingestion_metadata: Metadata from document ingestion
        execution_timestamp: Timestamp of workflow execution
        
    Returns:
        Dictionary containing structured report data
    """
    return {
        "metadata": {
            "execution_timestamp": execution_timestamp,
            "workflow_version": "1.0",
            "ingestion": ingestion_metadata
        },
        "summary": {
            "text": summary,
            "length": len(summary)
        },
        "decision": {
            "text": decision,
            "length": len(decision)
        },
        "report": {
            "text": reporter_output,
            "length": len(reporter_output),
            "word_count": len(reporter_output.split())
        }
    }


def run_report_generation(
    summary: str,
    decision: str,
    reporter_output: str,
    ingestion_metadata: Dict[str, Any],
    output_dir: Path = Path("."),
    output_filename: str = "report.txt",
    json_filename: str = "report.json"
) -> ReportGenerationResult:
    """
    Run the report generation agent to compile final structured report.
    
    Args:
        summary: Summary from summarizer agent
        decision: Decision from decision agent
        reporter_output: Report from reporter agent
        ingestion_metadata: Metadata from document ingestion
        output_dir: Directory to save the report
        output_filename: Name of the text report file
        json_filename: Name of the JSON report file
        
    Returns:
        ReportGenerationResult with report paths and metadata
    """
    logger.info("="*60)
    logger.info("REPORT GENERATION AGENT")
    logger.info("="*60)
    logger.info(f"System Prompt: {REPORT_GENERATION_SYSTEM_PROMPT[:100]}...")
    
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        execution_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create structured report
        logger.info("Compiling structured report...")
        report_content = create_structured_report(
            summary=summary,
            decision=decision,
            reporter_output=reporter_output,
            ingestion_metadata=ingestion_metadata,
            execution_timestamp=execution_timestamp
        )
        
        # Save text report
        report_file = output_path / output_filename
        report_file.write_text(report_content, encoding="utf-8")
        logger.info(f"Text report saved to: {report_file}")
        
        # Create JSON report
        logger.info("Creating JSON report...")
        json_report = create_json_report(
            summary=summary,
            decision=decision,
            reporter_output=reporter_output,
            ingestion_metadata=ingestion_metadata,
            execution_timestamp=execution_timestamp
        )
        
        # Save JSON report
        json_file = output_path / json_filename
        json_file.write_text(json.dumps(json_report, indent=2), encoding="utf-8")
        logger.info(f"JSON report saved to: {json_file}")
        
        # Prepare metadata
        metadata = {
            "execution_timestamp": execution_timestamp,
            "report_file": str(report_file),
            "json_file": str(json_file),
            "report_length": len(report_content),
            **ingestion_metadata
        }
        
        logger.info("Report generation completed successfully")
        logger.info(f"  Report length: {len(report_content)} characters")
        logger.info(f"  Files created: {output_filename}, {json_filename}")
        
        return ReportGenerationResult(
            report_path=str(report_file),
            json_path=str(json_file),
            report_content=report_content,
            metadata=metadata,
            success=True
        )
        
    except Exception as e:
        logger.exception(f"Error in report generation agent: {e}")
        return ReportGenerationResult(
            report_path="",
            json_path="",
            report_content="",
            metadata={},
            success=False,
            error_message=str(e)
        )

