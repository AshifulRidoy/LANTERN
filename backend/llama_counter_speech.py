import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModelForCausalLM,prepare_model_for_kbit_training
from datasets import Dataset
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class LLaMA3CounterSpeechGenerator:
    """
    LLaMA 3 model fine-tuned with LoRA for counter-speech generation.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        use_quantization: bool = True,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self.use_quantization = use_quantization
        
        # Configure quantization
        if use_quantization:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            self.bnb_config = None
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self._load_model()
        
        # Chat template for structured conversations
        self.system_prompt = """You are a compassionate and wise assistant who helps people respond constructively to hate speech. Your responses should be:
1. Empathetic and understanding
2. Non-confrontational but firm
3. Educational when appropriate
4. Focused on shared humanity
5. Never aggressive or hostile

Generate thoughtful counter-speech that promotes understanding and reduces conflict."""
    
    def _load_model(self):
        """Load the base model with optional quantization."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="eager",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
    
        
        logger.info("Model loaded successfully")
    
    def setup_lora(
        self,
        r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None
    ) -> None:
        """
        Setup LoRA configuration for efficient fine-tuning.
        
        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA
        """
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        if self.use_quantization:
           
            self.model = prepare_model_for_kbit_training(self.model)

        self.model.config.use_cache = False
        # Enable gradient checkpointing after preparation
        #self.model.gradient_checkpointing_enable()

        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, self.lora_config)
        #self.model = PeftModelForCausalLM(peft_model.base_model)
        self.model.train()
    
        # Enable gradient computation for LoRA parameters
        for param in self.model.parameters():
            if param.requires_grad:
                param.requires_grad_(True)
        self.model.print_trainable_parameters()
        

        logger.info("LoRA configuration applied")
    
    def format_chat_prompt(
        self,
        hate_speech: str,
        rationale: str = None,
        sentiment: str = None,
        context: str = None
    ) -> str:
        """
        Format input for chat-based generation.
        
        Args:
            hate_speech: The hate speech to respond to
            rationale: Explanation of why it's problematic
            sentiment: Detected sentiment
            context: Additional context
            
        Returns:
            Formatted chat prompt
        """
        user_message = f"Generate a respectful counter-speech response to this message: '{hate_speech}'"
        
        if rationale:
            user_message += f"\n\nThis message is problematic because: {rationale}"
        
        if sentiment:
            user_message += f"\nDetected sentiment: {sentiment}"
        
        if context:
            user_message += f"\nAdditional context: {context}"
        
        user_message += "\n\nPlease provide a calm, constructive response that promotes understanding."
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate_counter_speech(
        self,
        hate_speech: str,
        rationale: str = None,
        sentiment: str = None,
        context: str = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Generate counter-speech for given hate speech.
        
        Args:
            hate_speech: Input hate speech
            rationale: Explanation from rationale agent
            sentiment: Detected sentiment
            context: Additional context
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Dictionary containing generated counter-speech and metadata
        """
        # Format the prompt
        prompt = self.format_chat_prompt(
            hate_speech=hate_speech,
            rationale=rationale,
            sentiment=sentiment,
            context=context
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode the response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        ).strip()
        
        return {
            "original_hate_speech": hate_speech,
            "rationale": rationale,
            "sentiment": sentiment,
            "generated_counter_speech": generated_text,
            "prompt_used": prompt,
            "generation_params": {
                "temperature": temperature,
                "top_p": top_p,
                "max_length": max_length,
                "do_sample": do_sample
            }
        }
    
    def prepare_training_data(
        self,
        conan_data: List[Dict],
        additional_data: List[Dict] = None
    ) -> Dataset:
        """
        Prepare training data in chat format.
        
        Args:
            conan_data: CONAN dataset entries
            additional_data: Additional training data
            
        Returns:
            Formatted dataset for training
        """
        formatted_data = []
        
        # Process CONAN data
        for entry in conan_data:
            hate_speech = entry.get("hateSpeech", "")
            counter_speech = entry.get("counterSpeech", "")
            
            if hate_speech and counter_speech:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user", 
                        "content": f"Generate a respectful counter-speech response to this message: '{hate_speech}'\n\nPlease provide a calm, constructive response that promotes understanding."
                    },
                    {"role": "assistant", "content": counter_speech}
                ]
                
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False
                )
                
                formatted_data.append({"text": chat_text})
        
        # Process additional data if provided
        if additional_data:
            for entry in additional_data:
                if "messages" in entry:
                    chat_text = self.tokenizer.apply_chat_template(
                        entry["messages"],
                        tokenize=False
                    )
                    formatted_data.append({"text": chat_text})
        
        # Tokenize and prepare labels
        def tokenize(example):
            encoded = self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=1024
            )
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded

        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])
        print(tokenized_dataset[0])

        return tokenized_dataset
    
    def train(
        self,
        train_dataset: Dataset,
        output_dir: str = "./llama3-counter-speech-lora",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        max_grad_norm: float = 0.3,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "cosine"
    ):
        """
        Fine-tune the model using LoRA.
        
        Args are standard training arguments for the Trainer.
        """
        from trl import SFTTrainer
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim="adamw_torch",
            save_steps=500,
            logging_steps=25,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=max_grad_norm,
            max_steps=-1,
            warmup_ratio=warmup_ratio,
            group_by_length=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            lr_scheduler_type=lr_scheduler_type,
            report_to="wandb" if "wandb" in globals() else None,
        )
        
        trainer = SFTTrainer(
        model=self.model,
        train_dataset=train_dataset,
        peft_config=self.lora_config,  # Pass the LoRA config here
        # dataset_text_field="text",
        # max_seq_length=1024,
        # tokenizer=self.tokenizer,
        args=training_args,
        )

        self.model.train()

        # Start training
        trainer.train()
        
        # Save the model
        trainer.save_model()
        
        logger.info(f"Training completed. Model saved to {output_dir}")
    
    def load_trained_model(self, adapter_path: str):
        """
        Load a trained LoRA adapter.
        
        Args:
            adapter_path: Path to the trained LoRA adapter
        """
        logger.info(f"Loading trained adapter from {adapter_path}")
        
        # Load the PEFT model
        self.model = PeftModelForCausalLM.from_pretrained(
            self.model,
            adapter_path,
            torch_dtype=torch.bfloat16,
        )
        
        logger.info("Trained adapter loaded successfully")
    
    def save_model(self, save_path: str):
        """Save the current model state."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

class CounterSpeechValidator:
    """
    Validates generated counter-speech for safety and quality.
    """
    
    def __init__(self, hate_classifier, sentiment_classifier):
        self.hate_classifier = hate_classifier
        self.sentiment_classifier = sentiment_classifier
    
    def validate_response(self, counter_speech: str) -> Dict[str, Any]:
        """
        Validate a generated counter-speech response.
        
        Args:
            counter_speech: Generated counter-speech text
            
        Returns:
            Validation results
        """
        # Check if response contains hate speech
        hate_result = self.hate_classifier.predict(counter_speech)
        hate_score = hate_result.get('hate_probability', 0.0)
        is_hate = hate_result.get('hate_prediction', 0) == 1
        
        # Check sentiment of response
        sentiment_result = self.sentiment_classifier.predict(counter_speech)
        sentiment_class = sentiment_result.get('sentiment_prediction', 1)  # 0=neg, 1=neu, 2=pos
        sentiment_score = sentiment_result.get('sentiment_probability', 0.0)
        
        # Determine if response is safe
        is_safe = (not is_hate) and (sentiment_class >= 1)  # Not hate and neutral/positive
        
        return {
            "counter_speech": counter_speech,
            "is_safe": is_safe,
            "hate_score": hate_score,
            "is_hate": is_hate,
            "sentiment_class": sentiment_class,
            "sentiment_score": sentiment_score,
            "validation_passed": is_safe
        }

# Example usage
if __name__ == "__main__":
    # Initialize the counter-speech generator
    generator = LLaMA3CounterSpeechGenerator()
    
    # Setup LoRA for fine-tuning
    generator.setup_lora(r=64, lora_alpha=16, lora_dropout=0.1)
    
    # Example generation
    hate_text = "This group should not exist!"
    rationale = "This message promotes exclusion of a social group based on identity."
    
    result = generator.generate_counter_speech(
        hate_speech=hate_text,
        rationale=rationale,
        sentiment="negative"
    )
    
    print("Generated Counter-Speech:")
    print(result["generated_counter_speech"])
    print(f"\nOriginal: {result['original_hate_speech']}")
    print(f"Rationale: {result['rationale']}")
