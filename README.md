# Multi-Agent-Document-Processing-Using-Local-LLMs


A fully local Multiagent document Processing workflow using LangChain and Ollama, enabling multi-agent collaboration for document ingestion, summarization, decision making, and report generation. Integrated modular agent design for scalability and maintainability.

## 🌟 Features

- **Multi-Agent Pipeline**: Five specialized agents working in sequence
- **Document Processing**: Automatic ingestion, cleaning, and chunking of text documents
- **Intelligent Summarization**: Generate concise, bullet-point summaries
- **Decision Making**: Analyze content and determine if action is required
- **Report Generation**: Create natural language reports for stakeholders
- **Structured Output**: Both human-readable (TXT) and machine-readable (JSON) formats
- **Model Evaluation**: Compare different Ollama models for performance and quality
- **Flexible Configuration**: Customize models, temperature, and parameters per agent
- **Comprehensive Logging**: Detailed logs with rotating file handlers

## 📋 Workflow Overview

The system processes documents through a 5-step pipeline:

```
Document Ingestion → Summarization → Decision Making → Reporting → Report Generation
```

1. **Document Ingestion Agent**: Loads, cleans, and chunks text documents
2. **Summarizer Agent**: Generates concise bullet-point summaries
3. **Decision Agent**: Analyzes summaries and determines if action is required
4. **Reporter Agent**: Converts decisions into natural, readable explanations
5. **Report Generation Agent**: Compiles all outputs into structured reports

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- At least one Ollama model pulled (e.g., `phi3.5:latest`, `tinyllama:1.1b`)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Agentic Workflow"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama** (if not already running):
   ```bash
   ollama serve
   ```

5. **Pull a model** (if you haven't already):
   ```bash
   ollama pull phi3.5:latest
   # or
   ollama pull tinyllama:1.1b
   ```

### Basic Usage

1. **Place your documents** in the `artifacts/` directory (or specify a custom directory):
   ```bash
   # Example: Copy a text file
   cp your_document.txt artifacts/
   ```

2. **Run the pipeline**:
   ```bash
   python main.py
   ```

3. **Check the output**:
   - Text report: `report.txt`
   - JSON report: `report.json`
   - Logs: `logs/app.log` and `logs/exceptions.log`

## 📖 Detailed Usage

### Using the Main Script

```bash
python main.py [options]
```

**Options**:
- `--data-dir DIR`: Directory containing input text files (default: `artifacts/`)
- `--output-dir DIR`: Directory for output reports (default: current directory)
- `--summarizer-model MODEL`: Model for summarizer agent (default: `phi3.5:latest`)
- `--decision-model MODEL`: Model for decision agent (default: `phi3.5:latest`)
- `--reporter-model MODEL`: Model for reporter agent (default: `phi3.5:latest`)
- `--temperature FLOAT`: Temperature setting for all agents (default: `0.2`)
- `--report-filename NAME`: Name of the text report file (default: `report.txt`)
- `--json-report-filename NAME`: Name of the JSON report file (default: `report.json`)

**Example**:
```bash
python main.py --data-dir ./my_documents --summarizer-model llama3:8b --temperature 0.3
```

### Using the Pipeline Runner

The `run_pipeline.py` script provides a simplified interface:

```bash
python run_pipeline.py [options]
```

**Example**:
```bash
python run_pipeline.py --data-dir ./data --model phi3.5:latest --temperature 0.2
```

### Model Evaluation

Evaluate different Ollama models on the same workflow:

```bash
python evaluate_models.py [options]
```

**Options**:
- `--models MODEL1 MODEL2 ...`: List of models to evaluate (default: `tinyllama:1.1b llama3:8b mistral:7b`)
- `--data-dir DIR`: Directory containing input text files
- `--output FILE`: Output file for evaluation report (default: `evaluation_report.json`)
- `--temperature FLOAT`: Temperature for model inference (default: `0.2`)

**Example**:
```bash
python evaluate_models.py --models phi3.5:latest llama3:8b gemma:2b --output my_evaluation.json
```

## ⚙️ Configuration

### Environment Variables

You can configure the system using environment variables:

```bash
# Ollama Configuration
export OLLAMA_MODEL="phi3.5:latest"
export OLLAMA_BASE_URL="http://localhost:11434"

# Agent-specific Models
export SUMMARIZER_MODEL="phi3.5:latest"
export DECISION_MODEL="llama3:8b"
export REPORTER_MODEL="phi3.5:latest"

# Temperature
export TEMPERATURE="0.2"

# Directories
export DATA_DIR="artifacts/"
export OUTPUT_DIR="./"

# Output Files
export REPORT_FILENAME="report.txt"
export JSON_REPORT_FILENAME="report.json"
```

### Using a `.env` File

Create a `.env` file in the project root:

```env
OLLAMA_MODEL=phi3.5:latest
OLLAMA_BASE_URL=http://localhost:11434
SUMMARIZER_MODEL=phi3.5:latest
DECISION_MODEL=llama3:8b
REPORTER_MODEL=phi3.5:latest
TEMPERATURE=0.2
DATA_DIR=artifacts/
OUTPUT_DIR=./
REPORT_FILENAME=report.txt
JSON_REPORT_FILENAME=report.json
```

## 📁 Project Structure

```
Agentic Workflow/
├── agents/                      # Agent modules
│   ├── __init__.py
│   ├── doc_ingestion_agent.py   # Document ingestion agent
│   ├── summarizer_agent.py      # Summarization agent
│   ├── decision_agent.py        # Decision-making agent
│   ├── reporter_agent.py        # Reporting agent
│   └── report_generation_agent.py # Report generation agent
├── artifacts/                   # Input documents directory
│   └── sample.txt              # Example document
├── logs/                        # Log files
│   ├── app.log                 # General application logs
│   └── exceptions.log          # Exception logs
├── main.py                      # Main workflow controller
├── run_pipeline.py             # Simplified pipeline runner
├── evaluate_models.py          # Model evaluation script
├── logging_config.py           # Logging configuration
├── requirements.txt            # Python dependencies
├── report.txt                  # Output text report
├── report.json                 # Output JSON report
└── README.md                   # This file
```

## 🔧 Agent Details

### Document Ingestion Agent

- **Purpose**: Loads, cleans, and chunks text documents
- **Input**: Directory containing text files
- **Output**: Combined and cleaned text with metadata
- **Features**:
  - Automatic text cleaning
  - Configurable chunking (size and overlap)
  - Metadata collection (file names, character counts, etc.)

### Summarizer Agent

- **Purpose**: Generates concise bullet-point summaries
- **Input**: Combined text from ingestion
- **Output**: Structured summary with bullet points
- **Features**:
  - Configurable number of bullet points
  - Temperature control for creativity
  - Automatic bullet point counting

### Decision Agent

- **Purpose**: Analyzes summaries and makes decisions
- **Input**: Summary text
- **Output**: Decision with action required flag, justification, and next steps
- **Features**:
  - Action required detection
  - Justification extraction
  - Next steps identification

### Reporter Agent

- **Purpose**: Converts decisions into natural language reports
- **Input**: Summary and decision text
- **Output**: Natural language report
- **Features**:
  - Word count limits
  - Stakeholder-friendly language
  - Professional formatting

### Report Generation Agent

- **Purpose**: Compiles all outputs into structured reports
- **Input**: All previous agent outputs
- **Output**: TXT and JSON report files
- **Features**:
  - Human-readable text format
  - Machine-readable JSON format
  - Metadata inclusion
  - Timestamp tracking

## 📊 Output Format

### Text Report (`report.txt`)

A human-readable report containing:
- Executive summary
- Document ingestion metadata
- Generated summary
- Decision and analysis
- Natural language report
- Timestamp and workflow version

### JSON Report (`report.json`)

A structured JSON file containing:
- Metadata (execution timestamp, workflow version)
- Ingestion metadata (document counts, chunking info)
- Summary (text and length)
- Decision (text, action required, next steps)
- Report (text, length, word count)

## 🧪 Examples

### Example 1: Basic Usage

```bash
# Place documents in artifacts/
echo "Your document content here" > artifacts/my_doc.txt

# Run the pipeline
python main.py

# Check outputs
cat report.txt
cat report.json
```

### Example 2: Custom Models

```bash
# Use different models for each agent
python main.py \
  --summarizer-model phi3.5:latest \
  --decision-model llama3:8b \
  --reporter-model mistral:7b
```

### Example 3: Custom Configuration

```bash
# Use custom directories and settings
python main.py \
  --data-dir ./my_documents \
  --output-dir ./results \
  --temperature 0.5 \
  --report-filename my_report.txt
```

### Example 4: Model Evaluation

```bash
# Evaluate multiple models
python evaluate_models.py \
  --models phi3.5:latest llama3:8b gemma:2b \
  --data-dir ./test_data \
  --output evaluation_results.json
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'logging_config'**
   - **Solution**: Ensure you're running scripts from the project root directory
   - The agents automatically add the parent directory to `sys.path`

2. **Ollama Connection Error**
   - **Solution**: Ensure Ollama is running: `ollama serve`
   - Check `OLLAMA_BASE_URL` environment variable (default: `http://localhost:11434`)

3. **Model Not Found**
   - **Solution**: Pull the model first: `ollama pull <model-name>`
   - Verify with: `ollama list`

4. **No Documents Found**
   - **Solution**: Ensure text files (`.txt`) are in the `artifacts/` directory
   - Or specify a custom directory with `--data-dir`

5. **Import Errors**
   - **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate the virtual environment if using one

### Logging

Check the log files for detailed error information:
- `logs/app.log`: General application logs
- `logs/exceptions.log`: Exception-specific logs

## 📦 Dependencies

- `langchain`: Core LangChain functionality
- `langchain-community`: Community integrations
- `langchain-core`: Core LangChain components
- `langchain-ollama`: Ollama integration for LangChain
- `python-dotenv`: Environment variable management
- `tiktoken`: Token counting utilities

## 🔒 License

[Specify your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📝 License

[Add license information here]

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for providing local LLM capabilities
- [LangChain](https://www.langchain.com/) for the agent framework
- All contributors and users of this project

---

**Note**: This project requires Ollama to be running locally. Make sure you have Ollama installed and at least one model pulled before running the pipeline.

