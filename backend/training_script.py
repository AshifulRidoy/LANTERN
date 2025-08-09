#!/usr/bin/env python3
"""
Complete training script for the Multi-Task Hate Speech Detection & Counter-Speech Generation system.
This script orchestrates the training of both the DeBERTa multi-task model and LLaMA 3 counter-speech generator.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    DebertaV2Config,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, f1_score

# Import our custom modules
from deberta_multitask import MultiTaskDeBERTa, MultiTaskTrainer
from llama_counter_speech import LLaMA3CounterSpeechGenerator, CounterSpeechValidator
from data_pipeline import DataPipelineManager, MultiTaskDataset
from langchain_agents import AgenticPipeline
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

token = os.getenv("HuggingFace_Token")

wandb.login(key=os.getenv("WANDB_API_KEY"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Model configurations
        self.deberta_model_name = "microsoft/deberta-v3-base"
        self.llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Training hyperparameters
        self.deberta_epochs = 3
        self.deberta_batch_size = 4
        self.deberta_learning_rate = 2e-5
        self.deberta_warmup_ratio = 0.1
        self.deberta_weight_decay = 0.01
        
        self.llama_epochs = 3
        self.llama_batch_size = 1
        self.llama_gradient_accumulation_steps = 8
        self.llama_learning_rate = 2e-4
        self.llama_warmup_ratio = 0.03
        
        # LoRA configuration
        self.lora_r = 64
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        
        # Data configuration
        self.max_sequence_length = 256
        self.max_samples_per_dataset = 10000
        self.test_size = 0.2
        self.val_size = 0.1
        
        # Output directories
        self.output_dir = "./outputs"
        self.deberta_output_dir = os.getenv("DEBERTA_MODEL_PATH")
        self.llama_output_dir = os.getenv("LLAMA_MODEL_PATH")
        
        # Logging and monitoring
        self.use_wandb = True
        self.logging_steps = 25
        self.eval_steps = 500
        self.save_steps = 1000
        
        # Hardware configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision = True
        self.gradient_checkpointing = True

class DeBERTaTrainingManager:
    """Manages the training of the multi-task DeBERTa model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.deberta_model_name,token=token)
        deberta_config = DebertaV2Config.from_pretrained(config.deberta_model_name)
        self.model = MultiTaskDeBERTa(deberta_config)
        
        # Initialize trainer
        self.trainer = MultiTaskTrainer(self.model, self.tokenizer, self.device)
        
        logger.info("DeBERTa training manager initialized")
    
    def prepare_data(self) -> tuple:
        """Prepare training, validation, and test datasets."""
        pipeline_manager = DataPipelineManager(self.tokenizer)
        
        train_dataset, val_dataset, test_dataset = pipeline_manager.create_detection_datasets(
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            max_samples=self.config.max_samples_per_dataset * 4  # 4 datasets
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.deberta_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.deberta_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.deberta_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, pipeline_manager
    
    def train(self) -> Dict[str, Any]:
        """Train the multi-task DeBERTa model."""
        logger.info("Starting DeBERTa training...")
        
        # Prepare data
        train_loader, val_loader, test_loader, pipeline_manager = self.prepare_data()
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.deberta_learning_rate,
            weight_decay=self.config.deberta_weight_decay
        )
        
        num_training_steps = len(train_loader) * self.config.deberta_epochs
        num_warmup_steps = int(num_training_steps * self.config.deberta_warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(self.config.deberta_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.deberta_epochs}")
            
            # Training phase
            self.model.train()
            train_losses = []
            
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for step, batch in enumerate(train_pbar):
                # Training step
                loss_dict = self.trainer.train_step(batch, optimizer)
                scheduler.step()
                
                train_losses.append(loss_dict)
                
                # Update progress bar
                avg_loss = np.mean([l['total_loss'] for l in train_losses[-100:]])
                train_pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Log to wandb
                if self.config.use_wandb and step % self.config.logging_steps == 0:
                    wandb.log({
                        'train/total_loss': loss_dict['total_loss'],
                        'train/hate_loss': loss_dict['hate_loss'],
                        'train/sentiment_loss': loss_dict['sentiment_loss'],
                        'train/rationale_loss': loss_dict['rationale_loss'],
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/step': step
                    })
            
            # Validation phase
            val_metrics = self.evaluate(val_loader, "validation")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_model(f"{self.config.deberta_output_dir}/best_model")
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            # Record training history
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': np.mean([l['total_loss'] for l in train_losses]),
                'train_hate_loss': np.mean([l['hate_loss'] for l in train_losses if l['hate_loss'] > 0]),
                'train_sentiment_loss': np.mean([l['sentiment_loss'] for l in train_losses if l['sentiment_loss'] > 0]),
                'val_loss': val_metrics['total_loss'],
                'val_hate_f1': val_metrics.get('hate_f1', 0.0),
                'val_sentiment_f1': val_metrics.get('sentiment_f1', 0.0)
            }
            training_history.append(epoch_history)
            
            logger.info(f"Epoch {epoch + 1} completed - Train Loss: {epoch_history['train_loss']:.4f}, Val Loss: {epoch_history['val_loss']:.4f}")
        
        # Final evaluation on test set
        test_metrics = self.evaluate(test_loader, "test")
        
        # Save final model
        self.save_model(f"{self.config.deberta_output_dir}/final_model")
        
        results = {
            'training_history': training_history,
            'test_metrics': test_metrics,
            'best_val_loss': best_val_loss,
            'model_path': f"{self.config.deberta_output_dir}/best_model"
        }
        
        logger.info("DeBERTa training completed!")
        return results
    
    def evaluate(self, dataloader: DataLoader, split: str = "validation") -> Dict[str, float]:
        """Evaluate the model on a given dataset."""
        self.model.eval()
        
        all_losses = []
        hate_predictions, hate_labels = [], []
        sentiment_predictions, sentiment_labels = [], []
        
        eval_pbar = tqdm(dataloader, desc=f"Evaluating {split}")
        
        with torch.no_grad():
            for batch in eval_pbar:
                batch = self.trainer.prepare_batch(batch)
                outputs = self.model(**batch)
                
                if outputs['loss'] is not None:
                    all_losses.append(outputs['loss'].item())
                
                # Collect predictions for metrics calculation
                if outputs['hate_logits'] is not None:
                    hate_preds = torch.argmax(outputs['hate_logits'], dim=-1).cpu().numpy()
                    hate_true = batch['hate_labels'].cpu().numpy()
                    
                    # Only include valid labels (not -100)
                    valid_mask = hate_true != -100
                    if valid_mask.any():
                        hate_predictions.extend(hate_preds[valid_mask])
                        hate_labels.extend(hate_true[valid_mask])
                
                if outputs['sentiment_logits'] is not None:
                    sentiment_preds = torch.argmax(outputs['sentiment_logits'], dim=-1).cpu().numpy()
                    sentiment_true = batch['sentiment_labels'].cpu().numpy()
                    
                    # Only include valid labels (not -100)
                    valid_mask = sentiment_true != -100
                    if valid_mask.any():
                        sentiment_predictions.extend(sentiment_preds[valid_mask])
                        sentiment_labels.extend(sentiment_true[valid_mask])
        
        # Calculate metrics
        metrics = {
            'total_loss': np.mean(all_losses) if all_losses else 0.0
        }
        
        if hate_predictions and hate_labels:
            hate_f1 = f1_score(hate_labels, hate_predictions, average='weighted')
            metrics['hate_f1'] = hate_f1
            
            hate_report = classification_report(
                hate_labels, hate_predictions,
                target_names=['not_hate', 'hate'],
                output_dict=True
            )
            metrics['hate_classification_report'] = hate_report
        
        if sentiment_predictions and sentiment_labels:
            sentiment_f1 = f1_score(sentiment_labels, sentiment_predictions, average='weighted')
            metrics['sentiment_f1'] = sentiment_f1
            
            sentiment_report = classification_report(
                sentiment_labels, sentiment_predictions,
                target_names=['negative', 'neutral', 'positive'],
                output_dict=True
            )
            metrics['sentiment_classification_report'] = sentiment_report
        
        # Log to wandb
        if self.config.use_wandb:
            wandb_metrics = {f'{split}/{k}': v for k, v in metrics.items() 
                           if isinstance(v, (int, float))}
            wandb.log(wandb_metrics)
        
        return metrics
    
    def save_model(self, save_path: str):
        """Save the trained model."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save training config
        config_dict = {
            'model_name': self.config.deberta_model_name,
            'num_hate_labels': 2,
            'num_sentiment_labels': 3,
            'max_length': self.config.max_sequence_length
        }
        
        with open(f"{save_path}/training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")

class LLaMATrainingManager:
    """Manages the training of the LLaMA 3 counter-speech generator."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize LLaMA generator
        self.generator = LLaMA3CounterSpeechGenerator(
            model_name=config.llama_model_name,
            use_quantization=True
        )
        
        # Setup LoRA
        self.generator.setup_lora(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        
        logger.info("LLaMA training manager initialized")
    
    def prepare_data(self):
        """Prepare counter-speech training data."""
        pipeline_manager = DataPipelineManager(self.generator.tokenizer)
        
        # Create generation dataset with rationale-aware prompts
        generation_dataset = pipeline_manager.create_generation_dataset(
            include_rationales=True
        )
        
        logger.info(f"Generation dataset prepared with {len(generation_dataset)} samples")
        return generation_dataset
    
    def train(self) -> Dict[str, Any]:
        """Train the LLaMA 3 counter-speech generator."""
        logger.info("Starting LLaMA 3 training...")
        
        # Prepare data
        train_dataset = self.prepare_data()
        
        # Train the model
        self.generator.train(
            train_dataset=train_dataset,
            output_dir=self.config.llama_output_dir,
            num_train_epochs=self.config.llama_epochs,
            per_device_train_batch_size=self.config.llama_batch_size,
            gradient_accumulation_steps=self.config.llama_gradient_accumulation_steps,
            learning_rate=self.config.llama_learning_rate,
            warmup_ratio=self.config.llama_warmup_ratio
        )
        
        results = {
            'model_path': self.config.llama_output_dir,
            'training_completed': True
        }
        
        logger.info("LLaMA 3 training completed!")
        return results

class EndToEndTrainer:
    """Orchestrates the complete training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.deberta_output_dir, exist_ok=True)
        os.makedirs(config.llama_output_dir, exist_ok=True)
        
        # Initialize training managers
        self.deberta_trainer = DeBERTaTrainingManager(config)
        self.llama_trainer = LLaMATrainingManager(config)
        
        # Initialize wandb if enabled
        if config.use_wandb:
            wandb.init(
                project="hate-speech-counter-speech",
                config=vars(config),
                name=f"multitask-training-{wandb.util.generate_id()}"
            )
    
    def train_complete_system(self) -> Dict[str, Any]:
        """Train the complete system end-to-end."""
        logger.info("Starting end-to-end training...")
        
        results = {}
        
        try:
            # Phase 1: Train DeBERTa multi-task model
            logger.info("=== Phase 1: Training DeBERTa Multi-Task Model ===")
            deberta_results = self.deberta_trainer.train()
            results['deberta'] = deberta_results
            
            # Phase 2: Train LLaMA 3 counter-speech generator
            logger.info("=== Phase 2: Training LLaMA 3 Counter-Speech Generator ===")
            llama_results = self.llama_trainer.train()
            results['llama'] = llama_results
            
            # Phase 3: Test complete agentic pipeline
            logger.info("=== Phase 3: Testing Agentic Pipeline ===")
            pipeline_results = self.test_agentic_pipeline(
                deberta_results['model_path'],
                llama_results['model_path']
            )
            results['pipeline'] = pipeline_results
            
            # Save complete results
            self.save_training_results(results)
            
            logger.info("End-to-end training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            results['error'] = str(e)
            raise
        
        finally:
            if self.config.use_wandb:
                wandb.finish()
        
        return results
    
    def test_agentic_pipeline(self, deberta_path: str, llama_path: str) -> Dict[str, Any]:
        """Test the complete agentic pipeline."""
        logger.info("Testing agentic pipeline...")
        
        try:
            # Load trained models
            from transformers import AutoTokenizer
            
            # Load DeBERTa
            deberta_tokenizer = AutoTokenizer.from_pretrained(deberta_path)
            deberta_config = DebertaV2Config.from_pretrained(deberta_path)
            deberta_model = MultiTaskDeBERTa.from_pretrained(deberta_path, config=deberta_config)
            deberta_trainer = MultiTaskTrainer(deberta_model, deberta_tokenizer)
            
            # Load LLaMA
            llama_generator = LLaMA3CounterSpeechGenerator()
            llama_generator.load_trained_model(llama_path)
            
            # Create validator
            validator = CounterSpeechValidator(deberta_trainer, deberta_trainer)
            
            # Initialize agentic pipeline
            pipeline = AgenticPipeline(
                deberta_trainer=deberta_trainer,
                llama_generator=llama_generator,
                counter_speech_validator=validator
            )
            
            # Test with sample inputs
            test_inputs = [
                "This group should not exist!",
                "They are all criminals and dangerous.",
                "These people don't belong here.",
                "I hate this community and their traditions.",
                "All of them are the same and cause problems."
            ]
            
            test_results = []
            
            for test_input in test_inputs:
                logger.info(f"Testing input: {test_input}")
                
                try:
                    result = pipeline.process(test_input)
                    test_results.append({
                        'input': test_input,
                        'success': True,
                        'result': result
                    })
                    
                    logger.info(f"Generated counter-speech: {result.get('counter_speech', 'N/A')}")
                    
                except Exception as e:
                    logger.error(f"Pipeline failed for input '{test_input}': {str(e)}")
                    test_results.append({
                        'input': test_input,
                        'success': False,
                        'error': str(e)
                    })
            
            # Calculate success rate
            success_rate = sum(1 for r in test_results if r['success']) / len(test_results)
            
            pipeline_results = {
                'test_results': test_results,
                'success_rate': success_rate,
                'total_tests': len(test_results),
                'successful_tests': sum(1 for r in test_results if r['success'])
            }
            
            logger.info(f"Pipeline testing completed. Success rate: {success_rate:.2%}")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline testing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_training_results(self, results: Dict[str, Any]):
        """Save complete training results."""
        results_path = f"{self.config.output_dir}/training_results.json"
        
        # Convert non-serializable objects to serializable format
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Training results saved to {results_path}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train hate speech detection and counter-speech generation system")
    
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--deberta-only", action="store_true", help="Train only DeBERTa model")
    parser.add_argument("--llama-only", action="store_true", help="Train only LLaMA model")
    parser.add_argument("--test-only", action="store_true", help="Test pipeline only")
    parser.add_argument("--wandb", action="store_true", default=True, help="Use Weights & Biases logging")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig()
    config.use_wandb = args.wandb
    config.output_dir = args.output_dir
    
    # Load custom config if provided
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
        
        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Initialize trainer
    trainer = EndToEndTrainer(config)
    
    try:
        if args.test_only:
            # Test existing models
            deberta_path = f"{config.deberta_output_dir}/best_model"
            llama_path = config.llama_output_dir
            
            if Path(deberta_path).exists() and Path(llama_path).exists():
                results = trainer.test_agentic_pipeline(deberta_path, llama_path)
                print("Pipeline test results:")
                print(json.dumps(results, indent=2))
            else:
                logger.error("Trained models not found. Please train the models first.")
        
        elif args.deberta_only:
            # Train only DeBERTa
            results = trainer.deberta_trainer.train()
            print("DeBERTa training completed:")
            print(json.dumps(results, indent=2, default=str))
        
        elif args.llama_only:
            # Train only LLaMA
            results = trainer.llama_trainer.train()
            print("LLaMA training completed:")
            print(json.dumps(results, indent=2, default=str))
        
        else:
            # Train complete system
            results = trainer.train_complete_system()
            print("Complete training finished:")
            print(f"DeBERTa model saved to: {results['deberta']['model_path']}")
            print(f"LLaMA model saved to: {results['llama']['model_path']}")
            print(f"Pipeline success rate: {results['pipeline']['success_rate']:.2%}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()