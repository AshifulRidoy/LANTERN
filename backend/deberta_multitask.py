import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class MultiTaskDeBERTa(DebertaV2PreTrainedModel):
    """
    Multi-task DeBERTa model for hate speech detection, sentiment analysis, and rationale extraction.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_hate_labels = 2  # hate/not hate
        self.num_sentiment_labels = 3  # negative/neutral/positive
        
        # Shared encoder
        self.deberta = DebertaV2Model(config)
        # Remove gradient checkpointing - it can cause graph issues
        # self.deberta.gradient_checkpointing_enable()
        
        # Task-specific heads
        self.hate_head = nn.Linear(config.hidden_size, self.num_hate_labels)
        self.sentiment_head = nn.Linear(config.hidden_size, self.num_sentiment_labels)
        # Fix rationale decoder - should output 2 classes (rationale/not rationale), not vocab_size
        self.rationale_head = nn.Linear(config.hidden_size, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        hate_labels: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        rationale_labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for multi-task learning.
        """
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        if hate_labels is not None:
            hate_labels = hate_labels.to(device)
        if sentiment_labels is not None:
            sentiment_labels = sentiment_labels.to(device)
        if rationale_labels is not None:
            rationale_labels = rationale_labels.to(device)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get encoder outputs
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        
        # Extract representations
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = sequence_output[:, 0]  # CLS token representation
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)
        
        # Task-specific predictions
        hate_logits = self.hate_head(pooled_output)
        sentiment_logits = self.sentiment_head(pooled_output)
        rationale_logits = self.rationale_head(sequence_output)
        
        # Calculate losses if labels are provided
        total_loss = None
        hate_loss = None
        sentiment_loss = None
        rationale_loss = None
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Store individual losses for tracking (detached)
        losses_to_combine = []
        
        if hate_labels is not None:
            # Only calculate loss for samples with valid hate labels
            valid_hate_mask = hate_labels != -100
            if valid_hate_mask.any():
                hate_loss = loss_fct(
                    hate_logits[valid_hate_mask], 
                    hate_labels[valid_hate_mask]
                )
                losses_to_combine.append(hate_loss)
        
        if sentiment_labels is not None:
            # Only calculate loss for samples with valid sentiment labels
            valid_sentiment_mask = sentiment_labels != -100
            if valid_sentiment_mask.any():
                sentiment_loss = loss_fct(
                    sentiment_logits[valid_sentiment_mask], 
                    sentiment_labels[valid_sentiment_mask]
                )
                losses_to_combine.append(sentiment_loss)
        
        if rationale_labels is not None:
            # Flatten for token-level classification
            # Make sure rationale_labels are binary (0 or 1, with -100 for ignore)
            rationale_loss = loss_fct(
                rationale_logits.view(-1, 2),
                rationale_labels.view(-1)
            )
            losses_to_combine.append(rationale_loss)
        
        # Combine losses with equal weighting
        if losses_to_combine:
            total_loss = sum(losses_to_combine) / len(losses_to_combine)
        else:
            total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        # Return detached individual losses to prevent graph issues
        result = {
            "loss": total_loss,
            "hate_loss": hate_loss.detach() if hate_loss is not None else None,
            "sentiment_loss": sentiment_loss.detach() if sentiment_loss is not None else None,
            "rationale_loss": rationale_loss.detach() if rationale_loss is not None else None,
            "hate_logits": hate_logits,
            "sentiment_logits": sentiment_logits,
            "rationale_logits": rationale_logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
        
        return result
    
    def extract_rationales(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
        method: str = "token_classification"
    ) -> Dict[str, Any]:
        """
        Extract rationales using different methods.
        """
        # Ensure all tensors are on the same device as the model
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        rationales = {}
        
        if method == "token_classification":
            # Use rationale head predictions (2 classes: rationale/not rationale)
            rationale_probs = torch.softmax(outputs["rationale_logits"], dim=-1)
            # Get rationale scores (probability of being a rationale token)
            rationale_scores = rationale_probs[:, :, 1]  # Index 1 for rationale class
            
            for i in range(input_ids.size(0)):  # Iterate over batch
                # Move tensors to CPU for processing
                ids_cpu = input_ids[i].cpu()
                scores_cpu = rationale_scores[i].cpu()
                mask_cpu = attention_mask[i].cpu()
                
                # Convert to tokens
                tokens = tokenizer.convert_ids_to_tokens(ids_cpu.tolist())
                valid_mask = mask_cpu.bool()
                
                # Filter out special tokens and padding
                valid_tokens = []
                valid_scores = []
                rationale_tokens = []
                original_tokens = []
                
                for j, (token, score, is_valid) in enumerate(zip(tokens, scores_cpu, valid_mask)):
                    if is_valid and token not in ['[PAD]', '[CLS]', '[SEP]']:
                        original_tokens.append(token)
                        valid_tokens.append(token)
                        valid_scores.append(float(score))
                        
                        # Check if this token is a rationale (threshold > 0.5)
                        if float(score) > 0.3:
                            rationale_tokens.append(token)
                
                rationales[f"sample_{i}"] = {
                    "original_tokens": original_tokens,
                    "tokens": valid_tokens,
                    "rationale_tokens": rationale_tokens,
                    "rationale_scores": valid_scores,
                    "rationale_mask": [score > 0.5 for score in valid_scores]
                }
        
        elif method == "attention":
            # Use attention weights as rationales
            if outputs["attentions"] is not None:
                # Average across heads and layers
                attention_weights = torch.stack(outputs["attentions"]).mean(dim=(0, 2))
                
                for i in range(input_ids.size(0)):  # Iterate over batch
                    # Move to CPU for processing
                    ids_cpu = input_ids[i].cpu()
                    attn_cpu = attention_weights[i].cpu()
                    mask_cpu = attention_mask[i].cpu()
                    
                    tokens = tokenizer.convert_ids_to_tokens(ids_cpu.tolist())
                    valid_mask = mask_cpu.bool()
                    
                    # Focus on CLS token attention to other tokens
                    cls_attention = attn_cpu[0][valid_mask].numpy()
                    valid_tokens = [t for t, m in zip(tokens, valid_mask) if m and t not in ['[PAD]', '[CLS]', '[SEP]']]
                    
                    rationales[f"sample_{i}"] = {
                        "original_tokens": valid_tokens,
                        "tokens": valid_tokens,
                        "rationale_tokens": [t for t, score in zip(valid_tokens, cls_attention) if score > 0.1],
                        "attention_scores": cls_attention.tolist()
                    }
        
        return rationales

class MultiTaskTrainer:
    """
    Custom trainer for multi-task DeBERTa model with fixed gradient handling.
    """
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.scaler = GradScaler()
    
    def prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for training with proper label masking.
        """
        # Move tensors to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Handle missing labels by setting to -100 (ignored in loss calculation)
        if "hate_labels" not in batch:
            batch["hate_labels"] = torch.full(
                (batch["input_ids"].size(0),), -100, dtype=torch.long, device=self.device
            )
        
        if "sentiment_labels" not in batch:
            batch["sentiment_labels"] = torch.full(
                (batch["input_ids"].size(0),), -100, dtype=torch.long, device=self.device
            )
        
        if "rationale_labels" not in batch:
            batch["rationale_labels"] = torch.full(
                batch["input_ids"].shape, -100, dtype=torch.long, device=self.device
            )
        
        return batch
    
    def train_step(self, batch: Dict[str, Any], optimizer) -> Dict[str, float]:
        """
        Single training step with fixed gradient handling.
        """
        self.model.train()
        optimizer.zero_grad()
        
        batch = self.prepare_batch(batch)
        
        try:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs["loss"]
            
            # Check for NaN/Inf before backward
            if not torch.isfinite(loss):
                logger.warning("Loss is NaN or Inf! Skipping this batch.")
                return {
                    "total_loss": 0.0,
                    "hate_loss": 0.0,
                    "sentiment_loss": 0.0,
                    "rationale_loss": 0.0,
                }
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Extract loss values AFTER backward pass to avoid graph issues
            loss_dict = {
                "total_loss": loss.item(),
                "hate_loss": outputs["hate_loss"].item() if outputs["hate_loss"] is not None else 0.0,
                "sentiment_loss": outputs["sentiment_loss"].item() if outputs["sentiment_loss"] is not None else 0.0,
                "rationale_loss": outputs["rationale_loss"].item() if outputs["rationale_loss"] is not None else 0.0,
            }
            
            # Clear cache to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return loss_dict
            
        except RuntimeError as e:
            logger.error(f"Runtime error in train_step: {e}")
            optimizer.zero_grad()  # Clear gradients
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "total_loss": 0.0,
                "hate_loss": 0.0,
                "sentiment_loss": 0.0,
                "rationale_loss": 0.0,
            }
    
    def evaluate_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single evaluation step.
        """
        self.model.eval()
        batch = self.prepare_batch(batch)
        
        with torch.no_grad():
            outputs = self.model(**batch)
            
            loss_dict = {
                "total_loss": outputs["loss"].item(),
                "hate_loss": outputs["hate_loss"].item() if outputs["hate_loss"] is not None else 0.0,
                "sentiment_loss": outputs["sentiment_loss"].item() if outputs["sentiment_loss"] is not None else 0.0,
                "rationale_loss": outputs["rationale_loss"].item() if outputs["rationale_loss"] is not None else 0.0,
            }
            
            return loss_dict
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make predictions on a single text sample.
        """
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get predictions
        hate_probs = torch.softmax(outputs["hate_logits"], dim=-1)
        sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
        
        hate_pred = torch.argmax(hate_probs, dim=-1).item()
        sentiment_pred = torch.argmax(sentiment_probs, dim=-1).item()
        
        # Extract rationales
        rationales = self.model.extract_rationales(
            inputs["input_ids"], 
            inputs["attention_mask"], 
            self.tokenizer
        )
        
        return {
            "text": text,
            "hate_prediction": hate_pred,
            "hate_probability": hate_probs[0][hate_pred].item(),
            "sentiment_prediction": sentiment_pred,
            "sentiment_probability": sentiment_probs[0][sentiment_pred].item(),
            "rationales": rationales.get("sample_0", {}),
            "confidence": min(
                hate_probs[0][hate_pred].item(),
                sentiment_probs[0][sentiment_pred].item()
            )
        }

# Example usage and testing
if __name__ == "__main__":
    from transformers import DebertaV2Tokenizer, DebertaV2Config
    
    # Initialize model and tokenizer
    model_name = "microsoft/deberta-v3-base"
    config = DebertaV2Config.from_pretrained(model_name)
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    
    # Create multi-task model
    model = MultiTaskDeBERTa(config)
    trainer = MultiTaskTrainer(model, tokenizer)
    
    # Example prediction
    sample_text = "This group should not exist!"
    result = trainer.predict(sample_text)
    
    print("Prediction Results:")
    print(f"Text: {result['text']}")
    print(f"Hate Prediction: {result['hate_prediction']} (confidence: {result['hate_probability']:.3f})")
    print(f"Sentiment Prediction: {result['sentiment_prediction']} (confidence: {result['sentiment_probability']:.3f})")
    print(f"Overall Confidence: {result['confidence']:.3f}")
    
    # Test training step with dummy data
    print("\nTesting training step...")
    dummy_batch = {
        'input_ids': torch.randint(0, 1000, (2, 128)),
        'attention_mask': torch.ones(2, 128),
        'hate_labels': torch.tensor([0, 1]),
        'sentiment_labels': torch.tensor([1, 2]),
        'rationale_labels': torch.randint(0, 2, (2, 128))
    }
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    loss_dict = trainer.train_step(dummy_batch, optimizer)
    print(f"Training step completed: {loss_dict}")