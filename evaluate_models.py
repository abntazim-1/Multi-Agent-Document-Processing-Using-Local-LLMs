"""
Model Evaluation Script
Evaluates different Ollama models on the agentic workflow pipeline.
Compares performance, quality, and other metrics across models.

Usage:
    # Evaluate default models (tinyllama:1.1b, llama3:8b, mistral:7b)
    python evaluate_models.py

    # Evaluate specific models
    python evaluate_models.py --models tinyllama:1.1b llama3:8b gemma:2b

    # Custom data directory and output file
    python evaluate_models.py --data-dir ./my_data --output my_evaluation.json

    # Adjust temperature
    python evaluate_models.py --temperature 0.5

Requirements:
    - Ollama daemon running locally
    - Models must be pulled in Ollama (e.g., ollama pull llama3:8b)
    - Same dependencies as main.py
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

from logging_config import get_logger
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Import agent functions from main.py
from main import (
    read_text_files,
    combine_documents,
    summarizer_agent,
    decision_agent,
    reporter_agent,
    DATA_DIR,
    OLLAMA_BASE_URL
)

logger = get_logger("evaluate_models")


@dataclass
class ModelMetrics:
    """Metrics collected for each model evaluation."""
    model_name: str
    summarizer_time: float
    decision_time: float
    reporter_time: float
    total_time: float
    summary_length: int
    decision_length: int
    report_length: int
    summary_text: str
    decision_text: str
    report_text: str
    success: bool
    error_message: Optional[str] = None
    timestamp: str = ""


class ModelEvaluator:
    """Evaluates multiple models on the same workflow."""
    
    def __init__(self, models: List[str], base_url: str = OLLAMA_BASE_URL, temperature: float = 0.2):
        """
        Initialize the evaluator.
        
        Args:
            models: List of model names to evaluate (e.g., ["tinyllama:1.1b", "llama3:8b"])
            base_url: Ollama base URL
            temperature: Temperature setting for all models
        """
        self.models = models
        self.base_url = base_url
        self.temperature = temperature
        self.results: List[ModelMetrics] = []
        self.input_text_length: int = 0
    
    def create_llm_client(self, model_name: str) -> ChatOllama:
        """Create an Ollama LLM client for the given model."""
        return ChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=self.temperature
        )
    
    def evaluate_model(self, model_name: str, input_text: str) -> ModelMetrics:
        """
        Evaluate a single model on the complete workflow.
        
        Args:
            model_name: Name of the model to evaluate
            input_text: Input text to process
            
        Returns:
            ModelMetrics object with results
        """
        logger.info(f"Evaluating model: {model_name}")
        metrics = ModelMetrics(
            model_name=model_name,
            summarizer_time=0.0,
            decision_time=0.0,
            reporter_time=0.0,
            total_time=0.0,
            summary_length=0,
            decision_length=0,
            report_length=0,
            summary_text="",
            decision_text="",
            report_text="",
            success=False,
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # Create LLM client for this model
            llm = self.create_llm_client(model_name)
            
            # Test connection by making a simple call
            logger.info(f"Testing connection to {model_name}...")
            test_response = llm.invoke("Say 'OK' if you can hear me.")
            logger.info(f"Model {model_name} is ready")
            
            total_start = time.time()
            
            # 1. Summarizer Agent
            logger.info(f"Running summarizer agent with {model_name}...")
            start = time.time()
            try:
                summary = summarizer_agent(llm, input_text)
                metrics.summarizer_time = time.time() - start
                metrics.summary_text = summary
                metrics.summary_length = len(summary)
                logger.info(f"Summary generated in {metrics.summarizer_time:.2f}s ({metrics.summary_length} chars)")
            except Exception as e:
                logger.error(f"Summarizer agent failed for {model_name}: {e}")
                metrics.error_message = f"Summarizer failed: {str(e)}"
                return metrics
            
            # 2. Decision Agent
            logger.info(f"Running decision agent with {model_name}...")
            start = time.time()
            try:
                decision = decision_agent(llm, summary)
                metrics.decision_time = time.time() - start
                metrics.decision_text = decision
                metrics.decision_length = len(decision)
                logger.info(f"Decision generated in {metrics.decision_time:.2f}s ({metrics.decision_length} chars)")
            except Exception as e:
                logger.error(f"Decision agent failed for {model_name}: {e}")
                metrics.error_message = f"Decision agent failed: {str(e)}"
                return metrics
            
            # 3. Reporter Agent
            logger.info(f"Running reporter agent with {model_name}...")
            start = time.time()
            try:
                report = reporter_agent(llm, summary, decision)
                metrics.reporter_time = time.time() - start
                metrics.report_text = report
                metrics.report_length = len(report)
                logger.info(f"Report generated in {metrics.reporter_time:.2f}s ({metrics.report_length} chars)")
            except Exception as e:
                logger.error(f"Reporter agent failed for {model_name}: {e}")
                metrics.error_message = f"Reporter agent failed: {str(e)}"
                return metrics
            
            metrics.total_time = time.time() - total_start
            metrics.success = True
            logger.info(f"Model {model_name} completed successfully in {metrics.total_time:.2f}s")
            
        except Exception as e:
            logger.exception(f"Error evaluating model {model_name}: {e}")
            metrics.error_message = str(e)
            metrics.success = False
        
        return metrics
    
    def run_evaluation(self, input_text: str) -> List[ModelMetrics]:
        """
        Run evaluation on all models.
        
        Args:
            input_text: Input text to process with each model
            
        Returns:
            List of ModelMetrics for each model
        """
        self.input_text_length = len(input_text)
        logger.info(f"Starting evaluation of {len(self.models)} models")
        logger.info(f"Models to evaluate: {', '.join(self.models)}")
        logger.info(f"Input text length: {self.input_text_length} characters")
        
        results = []
        for i, model in enumerate(self.models, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating model {i}/{len(self.models)}: {model}")
            logger.info(f"{'='*60}")
            
            metrics = self.evaluate_model(model, input_text)
            results.append(metrics)
            
            # Brief pause between models to avoid overwhelming Ollama
            if i < len(self.models):
                time.sleep(1)
        
        self.results = results
        return results
    
    def generate_report(self, output_file: str = "evaluation_report.json") -> str:
        """
        Generate a comparison report from evaluation results.
        
        Args:
            output_file: Path to save the JSON report
            
        Returns:
            Formatted text report
        """
        if not self.results:
            return "No evaluation results available."
        
        # Convert results to dictionaries for JSON serialization
        results_dict = [asdict(r) for r in self.results]
        
        # Save JSON report
        json_path = Path(output_file)
        json_path.write_text(json.dumps(results_dict, indent=2), encoding="utf-8")
        logger.info(f"JSON report saved to {json_path}")
        
        # Generate text report
        report_lines = [
            "="*80,
            "MODEL EVALUATION REPORT",
            "="*80,
            f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Models Evaluated: {len(self.models)}",
            f"Input Text Length: {self.input_text_length} characters",
            "",
            "SUMMARY",
            "-"*80,
        ]
        
        # Summary table
        report_lines.append(f"{'Model':<30} {'Status':<10} {'Total Time':<12} {'Summary':<10} {'Decision':<10} {'Report':<10}")
        report_lines.append("-"*80)
        
        for r in self.results:
            status = "✓ Success" if r.success else "✗ Failed"
            report_lines.append(
                f"{r.model_name:<30} {status:<10} {r.total_time:>10.2f}s "
                f"{r.summary_length:>8} {r.decision_length:>8} {r.report_length:>8}"
            )
        
        report_lines.append("")
        report_lines.append("DETAILED RESULTS")
        report_lines.append("="*80)
        
        # Detailed results for each model
        for r in self.results:
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"MODEL: {r.model_name}")
            report_lines.append(f"{'='*80}")
            report_lines.append(f"Status: {'SUCCESS' if r.success else 'FAILED'}")
            
            if not r.success:
                report_lines.append(f"Error: {r.error_message}")
                continue
            
            report_lines.append(f"\nTiming:")
            report_lines.append(f"  Summarizer: {r.summarizer_time:.2f}s")
            report_lines.append(f"  Decision:    {r.decision_time:.2f}s")
            report_lines.append(f"  Reporter:    {r.reporter_time:.2f}s")
            report_lines.append(f"  Total:       {r.total_time:.2f}s")
            
            report_lines.append(f"\nOutput Lengths:")
            report_lines.append(f"  Summary: {r.summary_length} characters")
            report_lines.append(f"  Decision: {r.decision_length} characters")
            report_lines.append(f"  Report: {r.report_length} characters")
            
            report_lines.append(f"\nSummary:")
            report_lines.append(f"  {r.summary_text[:200]}{'...' if len(r.summary_text) > 200 else ''}")
            
            report_lines.append(f"\nDecision:")
            report_lines.append(f"  {r.decision_text[:200]}{'...' if len(r.decision_text) > 200 else ''}")
            
            report_lines.append(f"\nReport:")
            report_lines.append(f"  {r.report_text}")
            report_lines.append("")
        
        # Comparison section
        successful_results = [r for r in self.results if r.success]
        if len(successful_results) > 1:
            report_lines.append("="*80)
            report_lines.append("COMPARISON")
            report_lines.append("="*80)
            
            fastest = min(successful_results, key=lambda x: x.total_time)
            slowest = max(successful_results, key=lambda x: x.total_time)
            
            report_lines.append(f"Fastest Model: {fastest.model_name} ({fastest.total_time:.2f}s)")
            report_lines.append(f"Slowest Model: {slowest.model_name} ({slowest.total_time:.2f}s)")
            report_lines.append(f"Speed Difference: {slowest.total_time / fastest.total_time:.2f}x")
            
            avg_time = sum(r.total_time for r in successful_results) / len(successful_results)
            report_lines.append(f"Average Time: {avg_time:.2f}s")
        
        report_text = "\n".join(report_lines)
        
        # Save text report
        txt_path = Path(output_file.replace(".json", ".txt"))
        txt_path.write_text(report_text, encoding="utf-8")
        logger.info(f"Text report saved to {txt_path}")
        
        return report_text


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate multiple Ollama models on the agentic workflow")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tinyllama:1.1b", "llama3:8b", "mistral:7b"],
        help="List of model names to evaluate (default: tinyllama:1.1b llama3:8b mistral:7b)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help=f"Directory containing input text files (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output file for evaluation report (default: evaluation_report.json)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for model inference (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("MODEL EVALUATION SCRIPT")
    logger.info("="*80)
    logger.info(f"Models to evaluate: {args.models}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Temperature: {args.temperature}")
    
    # Read input documents
    data_path = Path(args.data_dir)
    docs = read_text_files(data_path)
    
    if not docs:
        logger.error(f"No documents found in {data_path}. Please add .txt files to the data directory.")
        return
    
    # Combine documents
    input_text = combine_documents(docs)
    
    if not input_text:
        logger.error("No input text available for evaluation.")
        return
    
    logger.info(f"Input text length: {len(input_text)} characters")
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(
        models=args.models,
        temperature=args.temperature
    )
    
    results = evaluator.run_evaluation(input_text)
    
    # Generate report
    report = evaluator.generate_report(args.output)
    
    # Print report to console
    print("\n" + report)
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"\nEvaluation complete: {successful}/{len(results)} models succeeded")


if __name__ == "__main__":
    main()

