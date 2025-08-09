# LANTERN – LangChain Agents for Neutralizing Toxicity & Enabling Respectful Narratives

A comprehensive AI system that combines multi-task hate speech detection with explainable counter-speech generation using LangChain agents orchestration. This project implements DeBERTa for detection and LLaMA 3 for generation, coordinated through specialized agents.

## 🌟 Features

- **Multi-Task Detection**: Simultaneous hate speech detection, sentiment analysis, and rationale extraction using DeBERTa-v3
- **Counter-Speech Generation**: Context-aware, empathetic responses using fine-tuned LLaMA 3 with LoRA
- **Agentic Orchestration**: Six specialized LangChain agents working in coordination
- **Explainable AI**: Multiple explanation methods including rationales, attention visualization, and counterfactual analysis
- **Safety Validation**: Automated guardrails with retry mechanisms
- **Production-Ready API**: FastAPI deployment with batch processing and monitoring
- **Comprehensive Evaluation**: ERASER-compatible rationale evaluation and safety metrics

### Model Components

#### DeBERTa Multi-Task Model
- **Base**: `microsoft/deberta-v3-base`
- **Tasks**: Hate detection (binary), sentiment analysis (3-class), rationale extraction
- **Training**: Multi-task learning with -100 label masking for heterogeneous datasets

#### LLaMA 3 Counter-Speech Generator
- **Base**: `meta-llama/Llama-3.1-8B-Instruct`
- **Fine-tuning**: LoRA (r=64, α=16) with 4-bit quantization
- **Training Data**: CONAN dataset with rationale-aware prompting
- **Safety**: Automated validation and retry mechanisms

## 📊 Dataset Integration

### Detection Datasets

| Dataset | Purpose | Labels | Size | Integration |
|---------|---------|---------|------|-------------|
| `english-hate-speech-superset` | Hate detection | Binary (0/1) | Large | DetectorAgent |
| `tweet_eval:sentiment` | Sentiment analysis | 3-class | Medium | DetectorAgent |
| `hatexplain` | Hate + rationale | 3-class + spans | Medium | RationaleAgent |
| `toxigen` | Toxicity detection | Binary | Large | GuardrailAgent |

### Generation Datasets

| Dataset | Purpose | Format | Integration |
|---------|---------|---------|-------------|
| `CONAN` | Counter-speech | Instruction pairs | CounterSpeechAgent |
| `IHSD` | Implicit hate | Text + label | CounterSpeechAgent |
| `Custom Synthetic` | Rationale-aware prompts | Enhanced instructions | CounterSpeechAgent |

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/agentic-hate-speech-detection.git
cd agentic-hate-speech-detection
```

2. **Set up environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Install PyTorch with CUDA support** (if using GPU)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Training

#### Train Complete System
```bash
python training_script.py
```

#### Train Individual Components
```bash
# Train only DeBERTa
python training_script.py --deberta-only

# Train only LLaMA
python training_script.py --llama-only
```

#### Custom Configuration
```bash
python training_script.py --config config.json --output-dir ./custom_outputs
```

### Deployment

#### Start API Server
```bash
python deployment_api.py
```

## 📚 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/process` | POST | Complete pipeline processing |
| `/detect` | POST | Hate speech and sentiment detection only |
| `/explain` | POST | Detection with rationale extraction |
| `/generate` | POST | Counter-speech generation only |
| `/validate` | POST | Safety validation of text |
| `/batch` | POST | Batch processing (up to 50 texts) |
| `/models/info` | GET | Information about loaded models |
| `/stats` | GET | API usage statistics |

### Agent-Specific Endpoints

| Endpoint | Description |
|----------|-------------|
| `/agents/detect` | Call DetectorAgent directly |
| `/agents/rationale` | Call RationaleAgent directly |
| `/agents/generate` | Call CounterSpeechAgent directly |
| `/agents/validate` | Call GuardrailAgent directly |

📈 Roadmap

Multi-Method Explainability

**⚠️ Important Note**: This system is designed for research purposes and content moderation applications. Please ensure responsible use and consider the ethical implications of automated hate speech detection and counter-speech generation in your specific context.