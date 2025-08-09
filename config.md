## 🔧 Configuration

### Training Configuration

```python
class TrainingConfig:
    # Model configurations
    deberta_model_name = "microsoft/deberta-v3-base"
    llama_model_name = "meta-llama/Llama-3-8B-Instruct"
    
    # Training hyperparameters
    deberta_epochs = 3
    deberta_batch_size = 8
    deberta_learning_rate = 2e-5
    
    llama_epochs = 3
    llama_batch_size = 1
    llama_gradient_accumulation_steps = 8
    llama_learning_rate = 2e-4
    
    # LoRA configuration
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    
    # Data configuration
    max_sequence_length = 512
    max_samples_per_dataset = 10000
```

### Environment Variables

```bash
# Model paths
DEBERTA_MODEL_PATH=./outputs/deberta-multitask/best_model
LLAMA_MODEL_PATH=./outputs/llama3-counter-speech

# API configuration
HOST=0.0.0.0
PORT=8000
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30

# Logging
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=hate-speech-counter-speech