import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
token = os.getenv("HuggingFace_Token")

logger = logging.getLogger(__name__)

class MultiTaskDataset(Dataset):
    """
    Custom dataset for multi-task learning with heterogeneous labels.
    Handles datasets with overlapping and exclusive labels using -100 masking.
    """
    
    def __init__(
        self,
        texts: List[str],
        hate_labels: Optional[List[int]] = None,
        sentiment_labels: Optional[List[int]] = None,
        rationale_spans: Optional[List[List[Tuple[int, int]]]] = None,
        tokenizer: AutoTokenizer = None,
        max_length: int = 512,
        dataset_names: Optional[List[str]] = None
    ):
        self.texts = texts
        self.hate_labels = hate_labels or [-100] * len(texts)
        self.sentiment_labels = sentiment_labels or [-100] * len(texts)
        self.rationale_spans = rationale_spans or [[] for _ in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_names = dataset_names or ["unknown"] * len(texts)
        
        # Validate data consistency
        assert len(self.texts) == len(self.hate_labels) == len(self.sentiment_labels)
        
        logger.info(f"Created MultiTaskDataset with {len(self.texts)} samples")
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        hate_label = self.hate_labels[idx]
        sentiment_label = self.sentiment_labels[idx]
        rationale_span = self.rationale_spans[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Create rationale labels for token-level classification
        rationale_labels = self._create_rationale_labels(
            encoding['offset_mapping'].squeeze(),
            rationale_span,
            len(encoding['input_ids'].squeeze())
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'hate_labels': torch.tensor(hate_label, dtype=torch.long),
            'sentiment_labels': torch.tensor(sentiment_label, dtype=torch.long),
            'rationale_labels': rationale_labels
        }
    
    def _create_rationale_labels(
        self, 
        offset_mapping: torch.Tensor, 
        rationale_spans: List[Tuple[int, int]], 
        seq_length: int
    ) -> torch.Tensor:
        """Create token-level rationale labels from character spans."""
        rationale_labels = torch.full((seq_length,), 0, dtype=torch.long)  # 0 for non-rationale
        
        if not rationale_spans:
            return rationale_labels
        
        # Map character spans to token positions
        for start_char, end_char in rationale_spans:
            for i, (token_start, token_end) in enumerate(offset_mapping):
                # Skip special tokens (they have (0,0) mapping)
                if token_start == 0 and token_end == 0:
                    continue
                    
                # Check if token overlaps with rationale span
                if (token_start < end_char and token_end > start_char):
                    rationale_labels[i] = 1  # 1 for rationale token
        
        return rationale_labels

class DatasetIntegrator:
    """
    Integrates multiple hate speech and sentiment datasets into a unified format.
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.datasets = {}
        
    def load_english_hate_speech_superset(self, split: str = "train") -> Dict[str, List]:
        """Load the english-hate-speech-superset dataset."""
        try:
            dataset = load_dataset("manueltonneau/english-hate-speech-superset", split=split, token=token)
            
            texts = []
            hate_labels = []
            dataset_names = []
            
            for item in dataset:
                texts.append(item['text'])
                # Map labels to binary (0: not hate, 1: hate)
                hate_labels.append(int(item['labels']))
                dataset_names.append("english-hate-speech-superset")
            
            logger.info(f"Loaded {len(texts)} samples from english-hate-speech-superset")
            
            return {
                'texts': texts,
                'hate_labels': hate_labels,
                'sentiment_labels': [-100] * len(texts),  # No sentiment labels
                'rationale_spans': [[] for _ in texts],  # No rationale spans
                'dataset_names': dataset_names
            }
            
        except Exception as e:
            logger.error(f"Failed to load english-hate-speech-superset: {e}")
            return {'texts': [], 'hate_labels': [], 'sentiment_labels': [], 'rationale_spans': [], 'dataset_names': []}

    def load_hatexplain_local(self, split: str = "train", base_path: str = "HateXplain/Data") -> Dict[str, List]:
        """
        Load HatEXplain dataset from local JSON files.
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            base_path: Path to HateXplain data directory
            
        Returns:
            Dictionary containing processed data
        """
        try:
            data_file = os.path.join(base_path, "dataset.json")
            split_file = os.path.join(base_path, "post_id_divisions.json")

            # Check if files exist
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Dataset file not found: {data_file}")
            if not os.path.exists(split_file):
                raise FileNotFoundError(f"Split file not found: {split_file}")

            # Load all data and post ID splits
            with open(data_file, "r", encoding='utf-8') as f:
                all_data = json.load(f)
            with open(split_file, "r", encoding='utf-8') as f:
                split_ids = json.load(f)

            # Get post IDs for the requested split
            if split not in split_ids:
                raise ValueError(f"Split '{split}' not found. Available splits: {list(split_ids.keys())}")
            
            post_ids = split_ids[split]
            
            texts = []
            hate_labels = []
            rationale_spans = []
            dataset_names = []

            # Label mapping: HatEXplain uses 3-class classification
            label_map = {
                "hatespeech": 1,   # Hate speech
                "offensive": 1,    # Offensive (also considered hate for binary)
                "normal": 0        # Normal/not hate
            }

            for post_id in post_ids:
                if post_id not in all_data:
                    logger.warning(f"Post ID {post_id} not found in dataset")
                    continue
                    
                item = all_data[post_id]
                tokens = item["post_tokens"]
                text = " ".join(tokens)

                # Use majority vote from all annotators for more robust labels
                annotator_labels = [ann["label"].lower() for ann in item["annotators"]]
                label_counts = {label: annotator_labels.count(label) for label in set(annotator_labels)}
                majority_label = max(label_counts, key=label_counts.get)
                
                # Convert to binary hate label
                hate_label = label_map.get(majority_label, 0)

                # Process rationales - use majority vote or first annotator
                rationales = item.get("rationales", [])
                if rationales:
                    # Use first annotator's rationale for simplicity
                    # In practice, you might want to combine multiple annotators
                    rationale_tokens = rationales[0] if len(rationales) > 0 else []
                    rationale_char_spans = self._convert_token_rationales_to_char_spans(text, rationale_tokens)
                else:
                    rationale_char_spans = []

                texts.append(text)
                hate_labels.append(hate_label)
                rationale_spans.append(rationale_char_spans)
                dataset_names.append("hatexplain")

            logger.info(f"Loaded {len(texts)} HatEXplain samples for split: {split}")
            logger.info(f"Label distribution: {dict(pd.Series(hate_labels).value_counts())}")

            return {
                "texts": texts,
                "hate_labels": hate_labels,
                "sentiment_labels": [-100] * len(texts),  # No sentiment labels
                "rationale_spans": rationale_spans,
                "dataset_names": dataset_names
            }

        except Exception as e:
            logger.error(f"Failed to load HatEXplain from {base_path}: {e}")
            return {
                "texts": [],
                "hate_labels": [],
                "sentiment_labels": [],
                "rationale_spans": [],
                "dataset_names": []
            }

    def _convert_token_rationales_to_char_spans(self, text: str, rationale_tokens: List[int]) -> List[Tuple[int, int]]:
        """
        Convert token-level rationale annotations to character spans.
        
        Args:
            text: Original text
            rationale_tokens: List of 0s and 1s indicating rationale tokens
            
        Returns:
            List of (start_char, end_char) tuples
        """
        if not rationale_tokens:
            return []
            
        tokens = text.split()
        spans = []
        current_span_start = None
        current_pos = 0
        
        # Process each token and its rationale annotation
        for i, token in enumerate(tokens):
            token_start = current_pos
            token_end = current_pos + len(token)
            
            # Check if we have rationale annotation for this token
            is_rationale = i < len(rationale_tokens) and rationale_tokens[i] == 1
            
            if is_rationale:
                if current_span_start is None:
                    # Start new span
                    current_span_start = token_start
                current_span_end = token_end
            else:
                if current_span_start is not None:
                    # End current span
                    spans.append((current_span_start, current_span_end))
                    current_span_start = None
            
            # Move to next token (account for space)
            current_pos = token_end + 1
        
        # Handle case where text ends with a rationale token
        if current_span_start is not None:
            spans.append((current_span_start, current_span_end))
        
        return spans

    def load_hatexplain_huggingface(self, split: str = "train") -> Dict[str, List]:
        """
        Load HatEXplain dataset from HuggingFace (if available).
        """
        try:
            # Try to load from HuggingFace hub
            dataset = load_dataset("hatexplain", split=split, use_auth_token=token)
            
            texts = []
            hate_labels = []
            rationale_spans = []
            dataset_names = []
            
            for item in dataset:
                text = " ".join(item['post_tokens'])
                
                # Convert label to binary
                label = item['annotators'][0]['label']  # Use first annotator
                hate_label = 1 if label in ['hatespeech', 'offensive'] else 0
                
                # Process rationales
                rationales = item.get('rationales', [[]])[0]  # First annotator's rationales
                char_spans = self._convert_token_rationales_to_char_spans(text, rationales)
                
                texts.append(text)
                hate_labels.append(hate_label)
                rationale_spans.append(char_spans)
                dataset_names.append("hatexplain")
            
            logger.info(f"Loaded {len(texts)} HatEXplain samples from HuggingFace")
            
            return {
                "texts": texts,
                "hate_labels": hate_labels,
                "sentiment_labels": [-100] * len(texts),
                "rationale_spans": rationale_spans,
                "dataset_names": dataset_names
            }
            
        except Exception as e:
            logger.warning(f"Failed to load HatEXplain from HuggingFace: {e}")
            logger.info("Falling back to local loading method")
            return self.load_hatexplain_local(split)
    
    def load_toxigen(self, split: str = "train", sample_size: int = 10000) -> Dict[str, List]:
        """Load a sample from the ToxiGen dataset."""
        try:
            dataset = load_dataset("toxigen/toxigen-data", name="annotated", split=split)
            
            # Sample a subset for manageable size
            if len(dataset) > sample_size:
                dataset = dataset.shuffle(seed=42).select(range(sample_size))
            
            texts = []
            hate_labels = []
            dataset_names = []
            
            for item in dataset:
                texts.append(item['text'])
                # ToxiGen uses 'toxicity_ai' and 'toxicity_human' fields
                # Use majority vote or average for binary classification
                toxicity_score = item.get('toxicity_ai', 0.5)
                hate_labels.append(1 if toxicity_score > 0.5 else 0)
                dataset_names.append("toxigen")
            
            logger.info(f"Loaded {len(texts)} samples from ToxiGen")
            
            return {
                'texts': texts,
                'hate_labels': hate_labels,
                'sentiment_labels': [-100] * len(texts),  # No sentiment labels
                'rationale_spans': [[] for _ in texts],  # No rationale spans
                'dataset_names': dataset_names
            }
            
        except Exception as e:
            logger.error(f"Failed to load ToxiGen: {e}")
            return {'texts': [], 'hate_labels': [], 'sentiment_labels': [], 'rationale_spans': [], 'dataset_names': []}

    def load_tweet_eval_sentiment(self, split: str = "train") -> Dict[str, List]:
        """Load the tweet_eval sentiment dataset."""
        try:
            dataset = load_dataset("tweet_eval", "sentiment", split=split)
            
            texts = []
            sentiment_labels = []
            dataset_names = []
            
            for item in dataset:
                texts.append(item['text'])
                sentiment_labels.append(item['label'])  # 0: negative, 1: neutral, 2: positive
                dataset_names.append("tweet_eval_sentiment")
            
            logger.info(f"Loaded {len(texts)} samples from tweet_eval sentiment")
            
            return {
                'texts': texts,
                'hate_labels': [-100] * len(texts),  # No hate labels
                'sentiment_labels': sentiment_labels,
                'rationale_spans': [[] for _ in texts],  # No rationale spans
                'dataset_names': dataset_names
            }
            
        except Exception as e:
            logger.error(f"Failed to load tweet_eval sentiment: {e}")
            return {'texts': [], 'hate_labels': [], 'sentiment_labels': [], 'rationale_spans': [], 'dataset_names': []}
    
    def create_combined_dataset(
        self,
        include_datasets: List[str] = None,
        max_samples_per_dataset: int = None,
        hatexplain_path: str = "HateXplain/Data"
    ) -> MultiTaskDataset:
        """
        Create a combined multi-task dataset from multiple sources.
        
        Args:
            include_datasets: List of dataset names to include
            max_samples_per_dataset: Maximum samples per dataset
            hatexplain_path: Path to HateXplain data directory
            
        Returns:
            Combined MultiTaskDataset
        """
        if include_datasets is None:
            include_datasets = ["hate_speech", "sentiment", "hatexplain", "toxigen"]
        
        all_texts = []
        all_hate_labels = []
        all_sentiment_labels = []
        all_rationale_spans = []
        all_dataset_names = []
        
        # Load hate speech data
        if "hate_speech" in include_datasets:
            hate_data = self.load_english_hate_speech_superset()
            if max_samples_per_dataset and hate_data['texts']:
                n_samples = min(len(hate_data['texts']), max_samples_per_dataset)
                for key in hate_data:
                    hate_data[key] = hate_data[key][:n_samples]
            
            all_texts.extend(hate_data['texts'])
            all_hate_labels.extend(hate_data['hate_labels'])
            all_sentiment_labels.extend(hate_data['sentiment_labels'])
            all_rationale_spans.extend(hate_data['rationale_spans'])
            all_dataset_names.extend(hate_data['dataset_names'])
        
        # Load sentiment data
        if "sentiment" in include_datasets:
            sentiment_data = self.load_tweet_eval_sentiment()
            if max_samples_per_dataset and sentiment_data['texts']:
                n_samples = min(len(sentiment_data['texts']), max_samples_per_dataset)
                for key in sentiment_data:
                    sentiment_data[key] = sentiment_data[key][:n_samples]
            
            all_texts.extend(sentiment_data['texts'])
            all_hate_labels.extend(sentiment_data['hate_labels'])
            all_sentiment_labels.extend(sentiment_data['sentiment_labels'])
            all_rationale_spans.extend(sentiment_data['rationale_spans'])
            all_dataset_names.extend(sentiment_data['dataset_names'])
        
        # Load HatEXplain data
        if "hatexplain" in include_datasets:
            # Try HuggingFace first, fall back to local
            hatexplain_data = self.load_hatexplain_huggingface()
            if not hatexplain_data['texts']:  # If HuggingFace failed
                hatexplain_data = self.load_hatexplain_local(base_path=hatexplain_path)
                
            if max_samples_per_dataset and hatexplain_data['texts']:
                n_samples = min(len(hatexplain_data['texts']), max_samples_per_dataset)
                for key in hatexplain_data:
                    hatexplain_data[key] = hatexplain_data[key][:n_samples]
            
            all_texts.extend(hatexplain_data['texts'])
            all_hate_labels.extend(hatexplain_data['hate_labels'])
            all_sentiment_labels.extend(hatexplain_data['sentiment_labels'])
            all_rationale_spans.extend(hatexplain_data['rationale_spans'])
            all_dataset_names.extend(hatexplain_data['dataset_names'])
        
        # Load ToxiGen data
        if "toxigen" in include_datasets:
            toxigen_data = self.load_toxigen()
            if max_samples_per_dataset and toxigen_data['texts']:
                n_samples = min(len(toxigen_data['texts']), max_samples_per_dataset)
                for key in toxigen_data:
                    toxigen_data[key] = toxigen_data[key][:n_samples]
            
            all_texts.extend(toxigen_data['texts'])
            all_hate_labels.extend(toxigen_data['hate_labels'])
            all_sentiment_labels.extend(toxigen_data['sentiment_labels'])
            all_rationale_spans.extend(toxigen_data['rationale_spans'])
            all_dataset_names.extend(toxigen_data['dataset_names'])
        
        logger.info(f"Combined dataset created with {len(all_texts)} total samples")
        if all_dataset_names:
            logger.info(f"Dataset composition: {dict(pd.Series(all_dataset_names).value_counts())}")
        
        return MultiTaskDataset(
            texts=all_texts,
            hate_labels=all_hate_labels,
            sentiment_labels=all_sentiment_labels,
            rationale_spans=all_rationale_spans,
            tokenizer=self.tokenizer,
            dataset_names=all_dataset_names
        )

class CounterSpeechDataProcessor:
    """
    Processes counter-speech generation datasets (CONAN, IHSD, etc.).
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def load_conan_dataset(self, file_path: str = None) -> List[Dict[str, str]]:
        """
        Load the CONAN counter-speech dataset.
        
        Args:
            file_path: Path to CONAN dataset file
            
        Returns:
            List of instruction-response pairs
        """
        # Since CONAN might not be directly available via HuggingFace,
        # this is a template for loading from local files
        try:
            if file_path and Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                # Fallback to synthetic examples for demonstration
                data = self._create_synthetic_conan_data()
            
            processed_data = []
            for item in data:
                if 'hateSpeech' in item and 'counterSpeech' in item:
                    processed_data.append({
                        'hate_speech': item['hateSpeech'],
                        'counter_speech': item['counterSpeech'],
                        'category': item.get('category', 'general')
                    })
            
            logger.info(f"Loaded {len(processed_data)} CONAN samples")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to load CONAN dataset: {e}")
            return self._create_synthetic_conan_data()
    
    def _create_synthetic_conan_data(self) -> List[Dict[str, str]]:
        """Create synthetic counter-speech training data."""
        return [
            {
                'hateSpeech': 'This group should not exist!',
                'counterSpeech': 'Every group deserves dignity and respect. Diversity makes our communities stronger.',
                'category': 'group_attack'
            },
            {
                'hateSpeech': 'They are all criminals and should be deported.',
                'counterSpeech': 'Generalizing about any group is unfair. Most people, regardless of background, are law-abiding and contribute positively to society.',
                'category': 'stereotyping'
            },
            {
                'hateSpeech': 'These people are destroying our culture.',
                'counterSpeech': 'Cultural exchange has always enriched societies. Different perspectives and traditions can coexist and learn from each other.',
                'category': 'cultural_attack'
            },
            {
                'hateSpeech': 'They don\'t belong here.',
                'counterSpeech': 'Everyone deserves to feel welcome and safe in their community. What matters is treating each other with kindness and respect.',
                'category': 'exclusion'
            },
            {
                'hateSpeech': 'All of them are the same.',
                'counterSpeech': 'Every person is unique with their own story, dreams, and contributions. It\'s important to see people as individuals.',
                'category': 'overgeneralization'
            }
        ]
    
    def format_for_instruction_tuning(
        self, 
        counter_speech_data: List[Dict[str, str]],
        system_prompt: str = None
    ) -> List[Dict[str, str]]:
        """
        Format counter-speech data for instruction tuning.
        
        Args:
            counter_speech_data: List of hate speech / counter-speech pairs
            system_prompt: System prompt for the model
            
        Returns:
            Formatted instruction data
        """
        if system_prompt is None:
            system_prompt = """You are a compassionate assistant who helps generate respectful responses to hate speech. Your responses should be empathetic, non-confrontational, and promote understanding."""
        
        formatted_data = []
        
        for item in counter_speech_data:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": f"Generate a respectful counter-speech response to this message: '{item['hate_speech']}'\n\nPlease provide a calm, constructive response that promotes understanding."
                },
                {"role": "assistant", "content": item['counter_speech']}
            ]
            
            # Apply chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False
            )
            
            formatted_data.append({
                'text': formatted_text,
                'original_hate': item['hate_speech'],
                'original_counter': item['counter_speech'],
                'category': item.get('category', 'general')
            })
        
        return formatted_data
    
    def create_rationale_aware_prompts(
        self, 
        counter_speech_data: List[Dict[str, str]],
        rationale_generator = None
    ) -> List[Dict[str, str]]:
        """
        Create rationale-aware prompts for counter-speech training.
        
        Args:
            counter_speech_data: Counter-speech data
            rationale_generator: Function to generate rationales
            
        Returns:
            Enhanced data with rationales
        """
        enhanced_data = []
        
        for item in counter_speech_data:
            hate_speech = item['hate_speech']
            counter_speech = item['counter_speech']
            
            # Generate or create rationale
            if rationale_generator:
                try:
                    rationale = rationale_generator(hate_speech)
                except:
                    rationale = "This message contains harmful language that targets individuals or groups."
            else:
                rationale = self._generate_simple_rationale(hate_speech)
            
            messages = [
                {"role": "system", "content": "You are a compassionate assistant who helps generate respectful responses to hate speech."},
                {
                    "role": "user", 
                    "content": f"Generate a respectful counter-speech response to this message: '{hate_speech}'\n\nThis message is problematic because: {rationale}\n\nPlease provide a calm, constructive response that promotes understanding."
                },
                {"role": "assistant", "content": counter_speech}
            ]
            
            formatted_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False
            )
            
            enhanced_data.append({
                'text': formatted_text,
                'rationale': rationale,
                'original_hate': hate_speech,
                'original_counter': counter_speech
            })
        
        return enhanced_data
    
    def _generate_simple_rationale(self, hate_speech: str) -> str:
        """Generate simple rationale based on keywords."""
        text_lower = hate_speech.lower()
        
        if any(word in text_lower for word in ['group', 'people', 'they', 'them']):
            return "This message targets or generalizes about a group of people."
        elif any(word in text_lower for word in ['not', 'should not', 'never']):
            return "This message contains exclusionary language."
        elif any(word in text_lower for word in ['all', 'every', 'always']):
            return "This message makes harmful generalizations."
        else:
            return "This message contains language that could be harmful or offensive."

class DataPipelineManager:
    """
    Main class for managing the complete data pipeline.
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.dataset_integrator = DatasetIntegrator(tokenizer)
        self.counter_speech_processor = CounterSpeechDataProcessor(tokenizer)
    
    def create_detection_datasets(
        self, 
        test_size: float = 0.2,
        val_size: float = 0.1,
        max_samples: int = None,
        hatexplain_path: str = "HateXplain/Data"
    ) -> Tuple[MultiTaskDataset, MultiTaskDataset, MultiTaskDataset]:
        """
        Create train/val/test datasets for detection tasks.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            max_samples: Maximum total samples
            hatexplain_path: Path to HateXplain data directory
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Create combined dataset
        combined_dataset = self.dataset_integrator.create_combined_dataset(
            max_samples_per_dataset=max_samples // 4 if max_samples else None,
            hatexplain_path=hatexplain_path
        )
        
        # Split into train/val/test
        indices = list(range(len(combined_dataset)))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=None
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_size/(1-test_size), random_state=42, stratify=None
        )
        
        # Create subset datasets
        train_dataset = self._create_subset_dataset(combined_dataset, train_indices)
        val_dataset = self._create_subset_dataset(combined_dataset, val_indices)
        test_dataset = self._create_subset_dataset(combined_dataset, test_indices)
        
        logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_generation_dataset(
        self, 
        conan_path: str = None,
        include_rationales: bool = True
    ) -> HFDataset:
        """
        Create dataset for counter-speech generation training.
        
        Args:
            conan_path: Path to CONAN dataset file
            include_rationales: Whether to include rationale-aware prompts
            
        Returns:
            HuggingFace Dataset for generation training
        """
        # Load counter-speech data
        conan_data = self.counter_speech_processor.load_conan_dataset(conan_path)
        
        # Format for instruction tuning
        if include_rationales:
            formatted_data = self.counter_speech_processor.create_rationale_aware_prompts(conan_data)
        else:
            formatted_data = self.counter_speech_processor.format_for_instruction_tuning(conan_data)
        
        # Convert to HuggingFace dataset
        dataset = HFDataset.from_list(formatted_data)
        
        logger.info(f"Created generation dataset with {len(dataset)} samples")
        return dataset
    
    def create_hatexplain_only_datasets(
        self,
        hatexplain_path: str = "HateXplain/Data",
        include_rationales: bool = True
    ) -> Tuple[MultiTaskDataset, MultiTaskDataset, MultiTaskDataset]:
        """
        Create train/val/test datasets using only HatEXplain data.
        This preserves the original HatEXplain splits.
        
        Args:
            hatexplain_path: Path to HateXplain data directory
            include_rationales: Whether to include rationale information
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_data = self.dataset_integrator.load_hatexplain_local("train", hatexplain_path)
        val_data = self.dataset_integrator.load_hatexplain_local("val", hatexplain_path)
        test_data = self.dataset_integrator.load_hatexplain_local("test", hatexplain_path)
        
        # Create datasets
        train_dataset = MultiTaskDataset(
            texts=train_data['texts'],
            hate_labels=train_data['hate_labels'],
            sentiment_labels=train_data['sentiment_labels'],
            rationale_spans=train_data['rationale_spans'] if include_rationales else None,
            tokenizer=self.tokenizer,
            dataset_names=train_data['dataset_names']
        )
        
        val_dataset = MultiTaskDataset(
            texts=val_data['texts'],
            hate_labels=val_data['hate_labels'],
            sentiment_labels=val_data['sentiment_labels'],
            rationale_spans=val_data['rationale_spans'] if include_rationales else None,
            tokenizer=self.tokenizer,
            dataset_names=val_data['dataset_names']
        )
        
        test_dataset = MultiTaskDataset(
            texts=test_data['texts'],
            hate_labels=test_data['hate_labels'],
            sentiment_labels=test_data['sentiment_labels'],
            rationale_spans=test_data['rationale_spans'] if include_rationales else None,
            tokenizer=self.tokenizer,
            dataset_names=test_data['dataset_names']
        )
        
        logger.info(f"Created HatEXplain-only datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset_dataset(self, original_dataset: MultiTaskDataset, indices: List[int]) -> MultiTaskDataset:
        """Create a subset of the original dataset."""
        subset_texts = [original_dataset.texts[i] for i in indices]
        subset_hate_labels = [original_dataset.hate_labels[i] for i in indices]
        subset_sentiment_labels = [original_dataset.sentiment_labels[i] for i in indices]
        subset_rationale_spans = [original_dataset.rationale_spans[i] for i in indices]
        subset_dataset_names = [original_dataset.dataset_names[i] for i in indices]
        
        return MultiTaskDataset(
            texts=subset_texts,
            hate_labels=subset_hate_labels,
            sentiment_labels=subset_sentiment_labels,
            rationale_spans=subset_rationale_spans,
            tokenizer=original_dataset.tokenizer,
            max_length=original_dataset.max_length,
            dataset_names=subset_dataset_names
        )
    
    def get_data_statistics(self, dataset: MultiTaskDataset) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        hate_labels = [label for label in dataset.hate_labels if label != -100]
        sentiment_labels = [label for label in dataset.sentiment_labels if label != -100]
        
        stats = {
            'total_samples': len(dataset),
            'hate_samples': len(hate_labels),
            'sentiment_samples': len(sentiment_labels),
            'hate_distribution': dict(pd.Series(hate_labels).value_counts()) if hate_labels else {},
            'sentiment_distribution': dict(pd.Series(sentiment_labels).value_counts()) if sentiment_labels else {},
            'dataset_composition': dict(pd.Series(dataset.dataset_names).value_counts()),
            'avg_text_length': np.mean([len(text.split()) for text in dataset.texts]),
            'rationale_samples': sum(1 for spans in dataset.rationale_spans if spans)
        }
        
        return stats

# Utility functions for HatEXplain setup
def setup_hatexplain_dataset(base_dir: str = "HateXplain"):
    """
    Helper function to set up HatEXplain dataset directory structure.
    
    Args:
        base_dir: Base directory to create the HateXplain folder
    """
    data_dir = os.path.join(base_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Created directory structure: {data_dir}")
    print("Please place the following files in the Data directory:")
    print("  - dataset.json (main dataset file)")
    print("  - post_id_divisions.json (train/val/test splits)")
    print("\nThese files can be obtained from:")
    print("https://github.com/hate-alert/HateXplain")

def validate_hatexplain_setup(base_path: str = "HateXplain/Data") -> bool:
    """
    Validate that HatEXplain dataset is properly set up.
    
    Args:
        base_path: Path to HateXplain data directory
        
    Returns:
        True if setup is valid, False otherwise
    """
    required_files = ["dataset.json", "post_id_divisions.json"]
    
    for file_name in required_files:
        file_path = os.path.join(base_path, file_name)
        if not os.path.exists(file_path):
            print(f"Missing required file: {file_path}")
            return False
        
        # Basic validation
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if file_name == "post_id_divisions.json":
                    required_splits = ["train", "val", "test"]
                    if not all(split in data for split in required_splits):
                        print(f"Missing splits in {file_name}. Expected: {required_splits}")
                        return False
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False
    
    print("HatEXplain dataset setup is valid!")
    return True

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", token=token)
    
    # Validate HatEXplain setup
    hatexplain_path = "HateXplain/Data"
    if not validate_hatexplain_setup(hatexplain_path):
        print("Setting up HatEXplain directory structure...")
        setup_hatexplain_dataset()
        print("Please download the dataset files and run again.")
        exit(1)
    
    # Create data pipeline manager
    pipeline_manager = DataPipelineManager(tokenizer)
    
    # Option 1: Create HateXplain-only datasets (preserves original splits)
    print("Creating HateXplain-only datasets...")
    train_dataset, val_dataset, test_dataset = pipeline_manager.create_hatexplain_only_datasets(
        hatexplain_path=hatexplain_path,
        include_rationales=True
    )
    
    # Print statistics
    print("\n=== HateXplain Train Dataset Statistics ===")
    train_stats = pipeline_manager.get_data_statistics(train_dataset)
    for key, value in train_stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== HateXplain Validation Dataset Statistics ===")
    val_stats = pipeline_manager.get_data_statistics(val_dataset)
    for key, value in val_stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== HateXplain Test Dataset Statistics ===")
    test_stats = pipeline_manager.get_data_statistics(test_dataset)
    for key, value in test_stats.items():
        print(f"  {key}: {value}")
    
    # Option 2: Create combined datasets (includes other datasets)
    print("\n" + "="*50)
    print("Creating combined datasets with HateXplain...")
    combined_train, combined_val, combined_test = pipeline_manager.create_detection_datasets(
        max_samples=2000,  # Small sample for testing
        hatexplain_path=hatexplain_path
    )
    
    # Print combined statistics
    print("\n=== Combined Train Dataset Statistics ===")
    combined_train_stats = pipeline_manager.get_data_statistics(combined_train)
    for key, value in combined_train_stats.items():
        print(f"  {key}: {value}")
    
    # Example of accessing a sample with rationales
    print("\n=== Sample Data Structure ===")
    sample = train_dataset[0]
    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Show rationale information if available
    if train_dataset.rationale_spans[0]:
        print(f"\nSample text: {train_dataset.texts[0]}")
        print(f"Rationale spans: {train_dataset.rationale_spans[0]}")
        print(f"Hate label: {train_dataset.hate_labels[0]}")
    
    print("\nHatEXplain integration complete! Dataset ready for training.")
    
    print("\nData pipeline ready for training!")
    
    