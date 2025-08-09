from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.llms import HuggingFacePipeline
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import json
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AgentResult(BaseModel):
    """Standardized result format for all agents."""
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None
    processing_time: float
    agent_name: str

class BaseAgent(ABC):
    """Base class for all specialized agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute the agent's main functionality."""
        pass
    
    def _time_execution(self, func, *args, **kwargs):
        """Helper to time execution of functions."""
        start_time = time.time()
        result = func(*args, **kwargs)
        processing_time = time.time() - start_time
        return result, processing_time

class DetectorAgent(BaseAgent):
    """Agent for running DeBERTa multi-task classification."""
    
    def __init__(self, deberta_trainer):
        super().__init__("DetectorAgent")
        self.deberta_trainer = deberta_trainer
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute hate speech and sentiment detection.
        
        Args:
            input_data: Dictionary containing 'text' key
            
        Returns:
            AgentResult with detection results
        """
        try:
            text = input_data.get('text', '')
            if not text:
                return AgentResult(
                    success=False,
                    data={},
                    error_message="No text provided for detection",
                    processing_time=0.0,
                    agent_name=self.name
                )
            
            # Execute detection with timing
            result, processing_time = self._time_execution(
                self.deberta_trainer.predict, text
            )
            
            self.call_count += 1
            
            return AgentResult(
                success=True,
                data={
                    'text': text,
                    'hate_prediction': result['hate_prediction'],
                    'hate_probability': result['hate_probability'],
                    'sentiment_prediction': result['sentiment_prediction'],
                    'sentiment_probability': result['sentiment_probability'],
                    'confidence': result['confidence'],
                    'call_count': self.call_count
                },
                processing_time=processing_time,
                agent_name=self.name
            )
            
        except Exception as e:
            logger.error(f"DetectorAgent failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                error_message=f"Detection failed: {str(e)}",
                processing_time=0.0,
                agent_name=self.name
            )

class RationaleAgent(BaseAgent):
    """Agent for extracting rationales from hate speech."""
    
    def __init__(self, deberta_trainer):
        super().__init__("RationaleAgent")
        self.deberta_trainer = deberta_trainer
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Extract rationales for detected hate speech.
        """
        try:
            text = input_data.get('text', '')
            hate_prediction = input_data.get('hate_prediction', 0)
            
            if not text:
                return AgentResult(
                    success=False,
                    data={},
                    error_message="No text provided for rationale extraction",
                    processing_time=0.0,
                    agent_name=self.name
                )
            
            # Only extract rationales if hate speech is detected
            if hate_prediction == 0:
                return AgentResult(
                    success=True,
                    data={
                        'text': text,
                        'rationale': "No hate speech detected, no rationale needed",
                        'token_level_rationale': [],
                        'explanation': "Content appears to be non-hateful"
                    },
                    processing_time=0.0,
                    agent_name=self.name
                )
            
            # Extract rationales with proper device handling
            tokenizer = self.deberta_trainer.tokenizer
            model = self.deberta_trainer.model
            
            # Tokenize (will be moved to correct device in extract_rationales)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

            # Ensure inputs are on the same device as the model
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            rationales, processing_time = self._time_execution(
                model.extract_rationales,
                inputs["input_ids"],
                inputs["attention_mask"],
                tokenizer,
                "token_classification"
            )
            
            # Generate natural language explanation
            sample_rationale = rationales.get("sample_0", {})
            original_tokens = sample_rationale.get("original_tokens", [])
            rationale_tokens = sample_rationale.get("rationale_tokens", [])
            
            print(f"DEBUG - Original tokens: {original_tokens}")
            print(f"DEBUG - Rationale tokens: {rationale_tokens}")
            print(f"DEBUG - Rationale scores: {sample_rationale.get('rationale_scores', [])}")
            # Create explanation based on rationale tokens
            explanation = self._generate_explanation(original_tokens, rationale_tokens, text)
            
            self.call_count += 1
            
            return AgentResult(
                success=True,
                data={
                    'text': text,
                    'rationale': explanation,
                    'token_level_rationale': rationale_tokens,
                    'original_tokens': original_tokens,
                    'explanation': explanation,
                    'call_count': self.call_count
                },
                processing_time=processing_time,
                agent_name=self.name
            )
            
        except Exception as e:
            logger.error(f"RationaleAgent failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                error_message=f"Rationale extraction failed: {str(e)}",
                processing_time=0.0,
                agent_name=self.name
            )

    def _ensure_device_consistency(self):
        """Ensure model and trainer are on the same device."""
        try:
            # Get the device from the model
            model_device = next(self.deberta_trainer.model.parameters()).device
            
            # Make sure the model is on the correct device
            self.deberta_trainer.model = self.deberta_trainer.model.to(model_device)
            
            return model_device
        except Exception as e:
            logger.warning(f"Could not determine model device: {e}")
            return torch.device('cpu')  # fallback to CPU        
        
    def _generate_explanation(self, original_tokens: List[str], rationale_tokens: List[str], text: str) -> str:
        """Generate natural language explanation from tokens."""
        if not rationale_tokens:
            return "Unable to identify specific problematic elements in the text."
        
        # Simple heuristic for generating explanations
        key_terms = [token for token in rationale_tokens if token not in ['[CLS]', '[SEP]', '[PAD]']]
        
        if "group" in key_terms or "people" in key_terms:
            return "The message targets or excludes a specific group of people."
        elif any(word in key_terms for word in ["not", "should", "never"]):
            return "The message contains exclusionary language or negative assertions."
        elif any(word in key_terms for word in ["hate", "bad", "wrong"]):
            return "The message contains explicitly negative or hateful language."
        else:
            return f"The message contains problematic elements: {', '.join(key_terms[:3])}"

class CounterSpeechAgent(BaseAgent):
    """Agent for generating counter-speech using LLaMA 3."""
    
    def __init__(self, llama_generator):
        super().__init__("CounterSpeechAgent")
        self.llama_generator = llama_generator
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Generate counter-speech for detected hate speech.
        
        Args:
            input_data: Dictionary containing text, rationale, and detection results
            
        Returns:
            AgentResult with generated counter-speech
        """
        try:
            text = input_data.get('text', '')
            rationale = input_data.get('rationale', '')
            sentiment_prediction = input_data.get('sentiment_prediction', 1)
            hate_prediction = input_data.get('hate_prediction', 0)
            
            if not text:
                return AgentResult(
                    success=False,
                    data={},
                    error_message="No text provided for counter-speech generation",
                    processing_time=0.0,
                    agent_name=self.name
                )
            
            # Only generate counter-speech for detected hate speech
            if hate_prediction == 0:
                return AgentResult(
                    success=True,
                    data={
                        'original_text': text,
                        'counter_speech': "No counter-speech needed for non-hateful content.",
                        'generation_skipped': True
                    },
                    processing_time=0.0,
                    agent_name=self.name
                )
            
            # Map sentiment prediction to label
            sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment = sentiment_labels.get(sentiment_prediction, "neutral")
            
            # Generate counter-speech with timing
            result, processing_time = self._time_execution(
                self.llama_generator.generate_counter_speech,
                hate_speech=text,
                rationale=rationale,
                sentiment=sentiment
            )
            
            self.call_count += 1
            
            return AgentResult(
                success=True,
                data={
                    'original_text': text,
                    'counter_speech': result['generated_counter_speech'],
                    'rationale_used': rationale,
                    'sentiment_used': sentiment,
                    'generation_params': result['generation_params'],
                    'call_count': self.call_count,
                    'generation_skipped': False
                },
                processing_time=processing_time,
                agent_name=self.name
            )
            
        except Exception as e:
            logger.error(f"CounterSpeechAgent failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                error_message=f"Counter-speech generation failed: {str(e)}",
                processing_time=0.0,
                agent_name=self.name
            )

class GuardrailAgent(BaseAgent):
    """Agent for validating generated counter-speech for safety."""
    
    def __init__(self, validator):
        super().__init__("GuardrailAgent")
        self.validator = validator
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Validate generated counter-speech for safety.
        
        Args:
            input_data: Dictionary containing counter-speech text
            
        Returns:
            AgentResult with validation results
        """
        try:
            counter_speech = input_data.get('counter_speech', '')
            generation_skipped = input_data.get('generation_skipped', False)
            
            if not counter_speech or generation_skipped:
                return AgentResult(
                    success=True,
                    data={
                        'counter_speech': counter_speech,
                        'is_safe': True,
                        'validation_skipped': True,
                        'reason': 'No counter-speech generated or generation was skipped'
                    },
                    processing_time=0.0,
                    agent_name=self.name
                )
            
            # Validate counter-speech with timing
            validation_result, processing_time = self._time_execution(
                self.validator.validate_response, counter_speech
            )
            
            self.call_count += 1
            
            return AgentResult(
                success=True,
                data={
                    'counter_speech': counter_speech,
                    'is_safe': validation_result['is_safe'],
                    'hate_score': validation_result['hate_score'],
                    'sentiment_class': validation_result['sentiment_class'],
                    'sentiment_score': validation_result['sentiment_score'],
                    'validation_passed': validation_result['validation_passed'],
                    'validation_skipped': False,
                    'call_count': self.call_count
                },
                processing_time=processing_time,
                agent_name=self.name
            )
            
        except Exception as e:
            logger.error(f"GuardrailAgent failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                error_message=f"Validation failed: {str(e)}",
                processing_time=0.0,
                agent_name=self.name
            )

class RetryAgent(BaseAgent):
    """Agent for handling regeneration with improved prompting."""
    
    def __init__(self, llama_generator, max_attempts: int = 2):
        super().__init__("RetryAgent")
        self.llama_generator = llama_generator
        self.max_attempts = max_attempts
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Retry counter-speech generation with improved prompting.
        
        Args:
            input_data: Dictionary containing original context and failure info
            
        Returns:
            AgentResult with retry results
        """
        try:
            original_text = input_data.get('original_text', '')
            rationale = input_data.get('rationale_used', '')
            sentiment = input_data.get('sentiment_used', 'neutral')
            attempt_count = input_data.get('attempt_count', 0)
            
            if attempt_count >= self.max_attempts:
                return AgentResult(
                    success=False,
                    data={
                        'original_text': original_text,
                        'counter_speech': None,
                        'max_attempts_reached': True,
                        'attempt_count': attempt_count
                    },
                    error_message=f"Maximum retry attempts ({self.max_attempts}) reached",
                    processing_time=0.0,
                    agent_name=self.name
                )
            
            # Enhance context for retry
            enhanced_context = self._create_enhanced_context(attempt_count, rationale)
            
            # Generate improved counter-speech with timing
            result, processing_time = self._time_execution(
                self.llama_generator.generate_counter_speech,
                hate_speech=original_text,
                rationale=rationale,
                sentiment=sentiment,
                context=enhanced_context,
                temperature=0.5,  # Lower temperature for more conservative generation
                top_p=0.8
            )
            
            self.call_count += 1
            
            return AgentResult(
                success=True,
                data={
                    'original_text': original_text,
                    'counter_speech': result['generated_counter_speech'],
                    'attempt_count': attempt_count + 1,
                    'enhanced_context': enhanced_context,
                    'generation_params': result['generation_params'],
                    'call_count': self.call_count
                },
                processing_time=processing_time,
                agent_name=self.name
            )
            
        except Exception as e:
            logger.error(f"RetryAgent failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                error_message=f"Retry generation failed: {str(e)}",
                processing_time=0.0,
                agent_name=self.name
            )
    
    def _create_enhanced_context(self, attempt_count: int, rationale: str) -> str:
        """Create enhanced context for retry attempts."""
        base_context = "Please ensure your response is completely safe, non-toxic, and promotes positive dialogue."
        
        if attempt_count == 1:
            return f"{base_context} Focus on empathy and understanding rather than direct confrontation."
        elif attempt_count >= 2:
            return f"{base_context} Use the gentlest possible approach, emphasizing shared humanity and common ground."
        
        return base_context

class DisplayAgent(BaseAgent):
    """Agent for formatting final output for frontend consumption."""
    
    def __init__(self):
        super().__init__("DisplayAgent")
    
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Format final output for UI consumption.
        
        Args:
            input_data: All previous agent results
            
        Returns:
            AgentResult with formatted display data
        """
        try:
            # Extract data from all previous agents
            formatted_output = {
                "original": input_data.get('original_text', ''),
                "detection": {
                    "hate_score": input_data.get('hate_probability', 0.0),
                    "sentiment": self._map_sentiment(input_data.get('sentiment_prediction', 1)),
                    "confidence": input_data.get('confidence', 0.0)
                },
                "rationale": {
                    "token_level": input_data.get('token_level_rationale', []),
                    "sentence_level": input_data.get('rationale', ''),
                    "explanation": input_data.get('explanation', '')
                },
                "counter_speech": input_data.get('counter_speech', ''),
                "validation": {
                    "is_safe": input_data.get('is_safe', False),
                    "hate_score": input_data.get('validation_hate_score', 0.0),
                    "sentiment": self._map_sentiment(input_data.get('validation_sentiment_class', 1)),
                    "attempts": input_data.get('attempt_count', 0)
                },
                "meta": {
                    "processing_time": sum([
                        input_data.get('detection_time', 0.0),
                        input_data.get('rationale_time', 0.0),
                        input_data.get('generation_time', 0.0),
                        input_data.get('validation_time', 0.0)
                    ]),
                    "agent_calls": sum([
                        input_data.get('detector_calls', 0),
                        input_data.get('rationale_calls', 0),
                        input_data.get('generation_calls', 0),
                        input_data.get('validation_calls', 0)
                    ]),
                    "retries": input_data.get('attempt_count', 0),
                    "model_versions": {
                        "deberta": "v3-base",
                        "llama": "3-8B-Instruct"
                    }
                }
            }
            
            self.call_count += 1
            
            return AgentResult(
                success=True,
                data=formatted_output,
                processing_time=0.0,  # Formatting is essentially instantaneous
                agent_name=self.name
            )
            
        except Exception as e:
            logger.error(f"DisplayAgent failed: {str(e)}")
            return AgentResult(
                success=False,
                data={},
                error_message=f"Output formatting failed: {str(e)}",
                processing_time=0.0,
                agent_name=self.name
            )
    
    def _map_sentiment(self, sentiment_code: int) -> str:
        """Map sentiment code to label."""
        mapping = {0: "negative", 1: "neutral", 2: "positive"}
        return mapping.get(sentiment_code, "neutral")

class AgenticPipeline:
    """Main orchestration class for the agentic pipeline."""
    
    def __init__(
        self,
        deberta_trainer,
        llama_generator,
        counter_speech_validator,
        max_retries: int = 2
    ):
        # Initialize all agents
        self.detector_agent = DetectorAgent(deberta_trainer)
        self.rationale_agent = RationaleAgent(deberta_trainer)
        self.counter_speech_agent = CounterSpeechAgent(llama_generator)
        self.guardrail_agent = GuardrailAgent(counter_speech_validator)
        self.retry_agent = RetryAgent(llama_generator, max_retries)
        self.display_agent = DisplayAgent()
        
        self.max_retries = max_retries
        
        logger.info("Agentic pipeline initialized with all agents")
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text through the complete agentic pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            Final formatted results
        """
        start_time = time.time()
        pipeline_data = {'text': text}
        
        try:
            # Step 1: Detection
            logger.info("Running DetectorAgent...")
            detection_result = self.detector_agent.execute(pipeline_data)
            if not detection_result.success:
                return self._create_error_response(detection_result.error_message)
            
            pipeline_data.update(detection_result.data)
            pipeline_data['detection_time'] = detection_result.processing_time
            pipeline_data['detector_calls'] = detection_result.data.get('call_count', 0)
            
            # Step 2: Rationale Extraction
            logger.info("Running RationaleAgent...")
            rationale_result = self.rationale_agent.execute(pipeline_data)
            if not rationale_result.success:
                return self._create_error_response(rationale_result.error_message)
            
            pipeline_data.update(rationale_result.data)
            pipeline_data['rationale_time'] = rationale_result.processing_time
            pipeline_data['rationale_calls'] = rationale_result.data.get('call_count', 0)
            
            # Step 3: Counter-Speech Generation
            logger.info("Running CounterSpeechAgent...")
            generation_result = self.counter_speech_agent.execute(pipeline_data)
            if not generation_result.success:
                return self._create_error_response(generation_result.error_message)
            
            pipeline_data.update(generation_result.data)
            pipeline_data['generation_time'] = generation_result.processing_time
            pipeline_data['generation_calls'] = generation_result.data.get('call_count', 0)
            
            # Step 4: Safety Validation with Retry Logic
            attempt_count = 0
            validation_passed = False
            
            while attempt_count <= self.max_retries and not validation_passed:
                logger.info(f"Running GuardrailAgent (attempt {attempt_count + 1})...")
                validation_result = self.guardrail_agent.execute(pipeline_data)
                
                if not validation_result.success:
                    return self._create_error_response(validation_result.error_message)
                
                validation_passed = validation_result.data.get('is_safe', False)
                pipeline_data['validation_time'] = validation_result.processing_time
                pipeline_data['validation_calls'] = validation_result.data.get('call_count', 0)
                
                # Update validation results
                pipeline_data['is_safe'] = validation_result.data.get('is_safe', False)
                pipeline_data['validation_hate_score'] = validation_result.data.get('hate_score', 0.0)
                pipeline_data['validation_sentiment_class'] = validation_result.data.get('sentiment_class', 1)
                
                if not validation_passed and attempt_count < self.max_retries:
                    logger.info(f"Validation failed, running RetryAgent (attempt {attempt_count + 1})...")
                    pipeline_data['attempt_count'] = attempt_count
                    retry_result = self.retry_agent.execute(pipeline_data)
                    
                    if retry_result.success:
                        # Update with new counter-speech
                        pipeline_data['counter_speech'] = retry_result.data.get('counter_speech')
                        pipeline_data['attempt_count'] = retry_result.data.get('attempt_count', 0)
                    
                attempt_count += 1
            
            # Step 5: Format Final Output
            logger.info("Running DisplayAgent...")
            display_result = self.display_agent.execute(pipeline_data)
            if not display_result.success:
                return self._create_error_response(display_result.error_message)
            
            # Add total processing time
            total_time = time.time() - start_time
            display_result.data['meta']['total_processing_time'] = total_time
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f}s")
            return display_result.data
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return self._create_error_response(f"Pipeline execution failed: {str(e)}")
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "success": False,
            "error": error_message,
            "original": "",
            "detection": {"hate_score": 0.0, "sentiment": "unknown", "confidence": 0.0},
            "rationale": {"explanation": "Error occurred during processing"},
            "counter_speech": "Unable to generate counter-speech due to error",
            "validation": {"is_safe": False, "attempts": 0},
            "meta": {"processing_time": 0.0, "agent_calls": 0, "retries": 0}
        }

# Example usage and testing
if __name__ == "__main__":
    # This would be initialized with actual trained models
    print("Agentic Pipeline Framework Ready")
    print("Agents implemented:")
    print("- DetectorAgent: Multi-task hate speech and sentiment detection")
    print("- RationaleAgent: Rationale extraction and explanation")
    print("- CounterSpeechAgent: LLaMA 3 counter-speech generation")
    print("- GuardrailAgent: Safety validation")
    print("- RetryAgent: Retry logic with improved prompting")
    print("- DisplayAgent: Output formatting")
    print("\nPipeline ready for integration with trained models.")