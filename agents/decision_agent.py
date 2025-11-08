"""
Decision Agent
Performs logical analysis and derives key insights or decisions.
"""

import os
from typing import Optional, List
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from logging_config import get_logger

logger = get_logger("decision_agent")

# System prompt for decision agent
DECISION_SYSTEM_PROMPT = """You are a Decision Agent specialized in logical analysis and strategic decision-making.

Your responsibilities include:
1. Analyzing summaries and extracting key insights
2. Determining whether action is required based on the content
3. Providing clear justifications for your decisions
4. Identifying next steps and recommendations
5. Assessing risks and opportunities

You make well-reasoned, logical decisions based on available information and provide actionable recommendations."""

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass
class DecisionResult:
    """Result from decision agent."""
    decision: str  # Decision text with analysis
    action_required: bool  # Whether action is needed
    justification: str  # Justification for the decision
    next_steps: List[str]  # List of recommended next steps
    decision_length: int  # Length of decision text in characters
    success: bool
    error_message: Optional[str] = None


def parse_decision_response(response_text: str) -> tuple:
    """
    Parse the decision response to extract action_required, justification, and next_steps.
    
    Args:
        response_text: Raw response from the decision agent
        
    Returns:
        Tuple of (action_required, justification, next_steps)
    """
    action_required = False
    justification = ""
    next_steps = []
    
    # Try to extract structured information
    text_lower = response_text.lower()
    
    # Check for action required indicators
    if any(indicator in text_lower for indicator in ["yes", "action required", "action needed", "requires action", "action is required"]):
        action_required = True
    elif any(indicator in text_lower for indicator in ["no", "no action", "no action needed", "not required", "action is not required"]):
        action_required = False
    
    # Try to extract next steps (look for numbered or bulleted lists)
    lines = response_text.split('\n')
    for line in lines:
        line_stripped = line.strip()
        # Look for numbered steps (1., 2., etc.) or bullet points
        if any(line_stripped.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*', '•']):
            # Clean up the step text
            step_text = line_stripped.lstrip('1234567890.-*• ').strip()
            if step_text:
                next_steps.append(step_text)
        elif 'next step' in text_lower or 'recommendation' in text_lower:
            line_lower = line_stripped.lower()
            # Extract text after keyword
            if ':' in line_stripped:
                step_text = line_stripped.split(':', 1)[1].strip()
                if step_text:
                    next_steps.append(step_text)
    
    # If we couldn't parse next steps, use a portion of the response
    if not next_steps:
        # Try to find justification section
        if 'justification' in text_lower or 'because' in text_lower or 'reason' in text_lower:
            justification = response_text[:500]  # Use first 500 chars as justification
        else:
            justification = response_text
    
    return action_required, justification, next_steps


def run_decision(
    summary: str,
    model_name: str = "tinyllama:1.1b",
    temperature: float = 0.2
) -> DecisionResult:
    """
    Run the decision agent to analyze summary and make decisions.
    
    Args:
        summary: Summary text from the summarizer agent
        model_name: Ollama model to use for decision-making
        temperature: Temperature setting for the model
        
    Returns:
        DecisionResult with decision, action_required flag, and next steps
    """
    logger.info("="*60)
    logger.info("DECISION AGENT")
    logger.info("="*60)
    logger.info(f"System Prompt: {DECISION_SYSTEM_PROMPT[:100]}...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Summary length: {len(summary)} characters")
    
    # Validate input
    if not summary or not summary.strip():
        error_msg = "Empty summary provided to decision agent"
        logger.error(error_msg)
        return DecisionResult(
            decision="",
            action_required=False,
            justification="",
            next_steps=[],
            decision_length=0,
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
            "You are a decision-making expert. Analyze the following summary and determine:\n"
            "1. Whether any action is required (Yes/No)\n"
            "2. A clear justification for your decision\n"
            "3. Two specific next steps or recommendations\n\n"
            "Summary:\n{summary}\n\n"
            "Provide your analysis in the following format:\n"
            "Action Required: [Yes/No]\n"
            "Justification: [Your reasoning]\n"
            "Next Steps:\n"
            "1. [First step]\n"
            "2. [Second step]"
        )
        
        # Invoke chain
        logger.info("Analyzing summary and making decision...")
        chain = prompt | llm
        response = chain.invoke({"summary": summary})
        
        decision_text = response.content.strip()
        
        # Parse the response
        action_required, justification, next_steps = parse_decision_response(decision_text)
        
        logger.info(f"Decision generated successfully ({len(decision_text)} characters)")
        logger.info(f"Action required: {action_required}")
        logger.info(f"Next steps identified: {len(next_steps)}")
        logger.debug(f"Decision preview: {decision_text[:200]}...")
        
        return DecisionResult(
            decision=decision_text,
            action_required=action_required,
            justification=justification or decision_text,
            next_steps=next_steps if next_steps else ["Review the summary for details"],
            decision_length=len(decision_text),
            success=True
        )
        
    except Exception as e:
        logger.exception(f"Error in decision agent: {e}")
        return DecisionResult(
            decision="",
            action_required=False,
            justification="",
            next_steps=[],
            decision_length=0,
            success=False,
            error_message=str(e)
        )

