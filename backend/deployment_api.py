#!/usr/bin/env python3
"""
FastAPI deployment server for the Agentic Hate Speech Detection & Counter-Speech Generation system.
Provides REST API endpoints for the complete pipeline and individual agents.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoTokenizer, DebertaV2Config

# Import our custom modules
from deberta_multitask import MultiTaskDeBERTa, MultiTaskTrainer
from llama_counter_speech import LLaMA3CounterSpeechGenerator, CounterSpeechValidator
from langchain_agents import AgenticPipeline
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model storage
models = {}

class ModelConfig:
    """Configuration for model paths and settings."""
    
    def __init__(self):
        self.deberta_model_path = os.getenv("DEBERTA_MODEL_PATH", "outputs/deberta-multitask/best_model")
        self.llama_model_path = os.getenv("LLAMA_MODEL_PATH", "outputs/llama3-counter-speech")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))

# Pydantic models for API request/response
class TextInput(BaseModel):
    """Input model for text processing."""
    text: str = Field(..., description="Input text to process", max_length=2000)
    include_rationale: bool = Field(True, description="Whether to include rationale extraction")
    generate_counter_speech: bool = Field(True, description="Whether to generate counter-speech")

class DetectionResult(BaseModel):
    """Detection result model."""
    hate_score: float = Field(..., description="Hate speech probability (0-1)")
    sentiment: str = Field(..., description="Sentiment classification")
    confidence: float = Field(..., description="Overall confidence score")

class RationaleResult(BaseModel):
    """Rationale extraction result model."""
    token_level: List[str] = Field(..., description="Token-level rationale")
    sentence_level: str = Field(..., description="Sentence-level explanation")
    explanation: str = Field(..., description="Natural language explanation")

class ValidationResult(BaseModel):
    """Validation result model."""
    is_safe: bool = Field(..., description="Whether the response is safe")
    hate_score: float = Field(..., description="Hate speech score of generated response")
    sentiment: str = Field(..., description="Sentiment of generated response")
    attempts: int = Field(..., description="Number of generation attempts")

class MetaInfo(BaseModel):
    """Metadata about processing."""
    processing_time: float = Field(..., description="Total processing time in seconds")
    agent_calls: int = Field(..., description="Number of agent calls made")
    retries: int = Field(..., description="Number of retry attempts")
    model_versions: Dict[str, str] = Field(..., description="Model versions used")

class PipelineResponse(BaseModel):
    """Complete pipeline response model."""
    success: bool = Field(..., description="Whether processing was successful")
    original: str = Field(..., description="Original input text")
    detection: DetectionResult = Field(..., description="Detection results")
    rationale: RationaleResult = Field(..., description="Rationale extraction results")
    counter_speech: str = Field(..., description="Generated counter-speech")
    validation: ValidationResult = Field(..., description="Validation results")
    meta: MetaInfo = Field(..., description="Processing metadata")
    error: Optional[str] = Field(None, description="Error message if any")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    version: str = Field(..., description="API version")
    uptime: float = Field(..., description="Service uptime in seconds")

class BatchRequest(BaseModel):
    """Batch processing request model."""
    texts: List[str] = Field(..., description="List of texts to process", max_items=50)
    include_rationale: bool = Field(True, description="Whether to include rationale extraction")
    generate_counter_speech: bool = Field(True, description="Whether to generate counter-speech")

class BatchResponse(BaseModel):
    """Batch processing response model."""
    results: List[PipelineResponse] = Field(..., description="Processing results for each input")
    total_processed: int = Field(..., description="Total number of texts processed")
    success_count: int = Field(..., description="Number of successful processings")
    total_processing_time: float = Field(..., description="Total processing time")

# Model loading and management
async def load_models():
    """Load all required models on startup."""
    config = ModelConfig()
    
    try:
        logger.info("Loading models...")
        
        # Load DeBERTa model
        if Path(config.deberta_model_path).exists():
            logger.info(f"Loading DeBERTa model from {config.deberta_model_path}")
            
            deberta_tokenizer = AutoTokenizer.from_pretrained(config.deberta_model_path)
            deberta_config = DebertaV2Config.from_pretrained(config.deberta_model_path)
            deberta_model = MultiTaskDeBERTa.from_pretrained(
                config.deberta_model_path, 
                config=deberta_config
            )
            deberta_trainer = MultiTaskTrainer(deberta_model, deberta_tokenizer, config.device)
            
            models['deberta_trainer'] = deberta_trainer
            logger.info("DeBERTa model loaded successfully")
        else:
            logger.error(f"DeBERTa model not found at {config.deberta_model_path}")
            models['deberta_trainer'] = None
        
        # Load LLaMA model
        if Path(config.llama_model_path).exists():
            logger.info(f"Loading LLaMA model from {config.llama_model_path}")
            
            llama_generator = LLaMA3CounterSpeechGenerator(use_quantization=True)
            llama_generator.load_trained_model(config.llama_model_path)
            
            models['llama_generator'] = llama_generator
            logger.info("LLaMA model loaded successfully")
        else:
            logger.error(f"LLaMA model not found at {config.llama_model_path}")
            models['llama_generator'] = None
        
        # Initialize validator
        if models['deberta_trainer']:
            validator = CounterSpeechValidator(
                models['deberta_trainer'], 
                models['deberta_trainer']
            )
            models['validator'] = validator
        else:
            models['validator'] = None
        
        # Initialize agentic pipeline
        if all(models[key] is not None for key in ['deberta_trainer', 'llama_generator', 'validator']):
            pipeline = AgenticPipeline(
                deberta_trainer=models['deberta_trainer'],
                llama_generator=models['llama_generator'],
                counter_speech_validator=models['validator']
            )
            models['pipeline'] = pipeline
            logger.info("Agentic pipeline initialized successfully")
        else:
            models['pipeline'] = None
            logger.error("Failed to initialize agentic pipeline - missing models")
        
        models['startup_time'] = time.time()
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load models on startup."""
    # Startup
    await load_models()
    yield
    # Shutdown
    logger.info("Shutting down application...")

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Hate Speech Detection & Counter-Speech API",
    description="REST API for multi-task hate speech detection and counter-speech generation using LangChain agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request rate limiting (simple in-memory counter)
request_counts = {}
REQUEST_LIMIT = 100  # requests per minute
WINDOW_SIZE = 60  # seconds

def get_client_ip(request) -> str:
    """Extract client IP from request."""
    return request.client.host

def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting check."""
    current_time = time.time()
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Clean old requests
    request_counts[client_ip] = [
        req_time for req_time in request_counts[client_ip] 
        if current_time - req_time < WINDOW_SIZE
    ]
    
    # Check limit
    if len(request_counts[client_ip]) >= REQUEST_LIMIT:
        return False
    
    request_counts[client_ip].append(current_time)
    return True

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - models.get('startup_time', time.time())
    
    return HealthResponse(
        status="healthy" if models.get('pipeline') is not None else "degraded",
        models_loaded={
            "deberta": models.get('deberta_trainer') is not None,
            "llama": models.get('llama_generator') is not None,
            "validator": models.get('validator') is not None,
            "pipeline": models.get('pipeline') is not None
        },
        version="1.0.0",
        uptime=uptime
    )

@app.post("/process", response_model=PipelineResponse)
async def process_text(request: TextInput):
    """Process text through the complete agentic pipeline."""
    pipeline = models.get('pipeline')
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not available - models not loaded")
    
    try:
        # Process through pipeline
        result = pipeline.process(request.text)
        
        # Convert to response format
        if result.get('success', True):  # Assuming success if not explicitly failed
            response = PipelineResponse(
                success=True,
                original=result.get('original', request.text),
                detection=DetectionResult(
                    hate_score=result['detection']['hate_score'],
                    sentiment=result['detection']['sentiment'],
                    confidence=result['detection']['confidence']
                ),
                rationale=RationaleResult(
                    token_level=result['rationale']['token_level'],
                    sentence_level=result['rationale']['sentence_level'],
                    explanation=result['rationale']['explanation']
                ),
                counter_speech=result.get('counter_speech', ''),
                validation=ValidationResult(
                    is_safe=result['validation']['is_safe'],
                    hate_score=result['validation']['hate_score'],
                    sentiment=result['validation']['sentiment'],
                    attempts=result['validation']['attempts']
                ),
                meta=MetaInfo(
                    processing_time=result['meta']['processing_time'],
                    agent_calls=result['meta']['agent_calls'],
                    retries=result['meta']['retries'],
                    model_versions=result['meta']['model_versions']
                )
            )
        else:
            # Handle error case
            response = PipelineResponse(
                success=False,
                original=request.text,
                detection=DetectionResult(hate_score=0.0, sentiment="unknown", confidence=0.0),
                rationale=RationaleResult(token_level=[], sentence_level="", explanation=""),
                counter_speech="",
                validation=ValidationResult(is_safe=False, hate_score=0.0, sentiment="unknown", attempts=0),
                meta=MetaInfo(processing_time=0.0, agent_calls=0, retries=0, model_versions={}),
                error=result.get('error', 'Unknown error occurred')
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/detect", response_model=Dict[str, Any])
async def detect_only(request: TextInput):
    """Run only hate speech and sentiment detection."""
    deberta_trainer = models.get('deberta_trainer')
    if deberta_trainer is None:
        raise HTTPException(status_code=503, detail="DeBERTa model not available")
    
    try:
        result = deberta_trainer.predict(request.text)
        return {
            "text": request.text,
            "hate_prediction": result['hate_prediction'],
            "hate_probability": result['hate_probability'],
            "sentiment_prediction": result['sentiment_prediction'],
            "sentiment_probability": result['sentiment_probability'],
            "confidence": result['confidence']
        }
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/explain", response_model=Dict[str, Any])
async def explain_detection(request: TextInput):
    """Run detection with rationale extraction."""
    deberta_trainer = models.get('deberta_trainer')
    if deberta_trainer is None:
        raise HTTPException(status_code=503, detail="DeBERTa model not available")
    
    try:
        # Get detection results
        detection_result = deberta_trainer.predict(request.text)
        
        # Extract rationales if hate speech is detected
        rationales = {}
        if detection_result['hate_prediction'] == 1:
            tokenizer = deberta_trainer.tokenizer
            inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
            device = next(deberta_trainer.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            rationales = deberta_trainer.model.extract_rationales(
                inputs["input_ids"],
                inputs["attention_mask"],
                tokenizer
            )
        
        return {
            "text": request.text,
            "detection": detection_result,
            "rationales": rationales.get("sample_0", {})
        }
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")

@app.post("/generate", response_model=Dict[str, Any])
async def generate_counter_speech(request: TextInput):
    """Generate counter-speech for given text."""
    llama_generator = models.get('llama_generator')
    if llama_generator is None:
        raise HTTPException(status_code=503, detail="LLaMA model not available")
    
    try:
        result = llama_generator.generate_counter_speech(
            hate_speech=request.text,
            rationale="Potentially harmful content detected" if request.include_rationale else None
        )
        return result
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/validate", response_model=Dict[str, Any])
async def validate_text(request: TextInput):
    """Validate text for safety."""
    validator = models.get('validator')
    if validator is None:
        raise HTTPException(status_code=503, detail="Validator not available")
    
    try:
        result = validator.validate_response(request.text)
        return result
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/batch", response_model=BatchResponse)
async def batch_process(request: BatchRequest, background_tasks: BackgroundTasks):
    """Process multiple texts in batch."""
    pipeline = models.get('pipeline')
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not available")
    
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Batch size too large (max 50)")
    
    try:
        start_time = time.time()
        results = []
        success_count = 0
        
        for text in request.texts:
            try:
                result = pipeline.process(text)
                
                # Convert to response format (simplified for batch)
                pipeline_response = PipelineResponse(
                    success=result.get('success', True),
                    original=result.get('original', text),
                    detection=DetectionResult(
                        hate_score=result.get('detection', {}).get('hate_score', 0.0),
                        sentiment=result.get('detection', {}).get('sentiment', 'unknown'),
                        confidence=result.get('detection', {}).get('confidence', 0.0)
                    ),
                    rationale=RationaleResult(
                        token_level=result.get('rationale', {}).get('token_level', []),
                        sentence_level=result.get('rationale', {}).get('sentence_level', ''),
                        explanation=result.get('rationale', {}).get('explanation', '')
                    ),
                    counter_speech=result.get('counter_speech', ''),
                    validation=ValidationResult(
                        is_safe=result.get('validation', {}).get('is_safe', False),
                        hate_score=result.get('validation', {}).get('hate_score', 0.0),
                        sentiment=result.get('validation', {}).get('sentiment', 'unknown'),
                        attempts=result.get('validation', {}).get('attempts', 0)
                    ),
                    meta=MetaInfo(
                        processing_time=result.get('meta', {}).get('processing_time', 0.0),
                        agent_calls=result.get('meta', {}).get('agent_calls', 0),
                        retries=result.get('meta', {}).get('retries', 0),
                        model_versions=result.get('meta', {}).get('model_versions', {})
                    )
                )
                
                results.append(pipeline_response)
                if pipeline_response.success:
                    success_count += 1
                    
            except Exception as e:
                logger.error(f"Batch processing failed for text '{text}': {str(e)}")
                # Add error result
                error_response = PipelineResponse(
                    success=False,
                    original=text,
                    detection=DetectionResult(hate_score=0.0, sentiment="unknown", confidence=0.0),
                    rationale=RationaleResult(token_level=[], sentence_level="", explanation=""),
                    counter_speech="",
                    validation=ValidationResult(is_safe=False, hate_score=0.0, sentiment="unknown", attempts=0),
                    meta=MetaInfo(processing_time=0.0, agent_calls=0, retries=0, model_versions={}),
                    error=str(e)
                )
                results.append(error_response)
        
        total_time = time.time() - start_time
        
        return BatchResponse(
            results=results,
            total_processed=len(request.texts),
            success_count=success_count,
            total_processing_time=total_time
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
@app.post("/initialize")
async def initialize_models():
    """Force model initialization if not already done."""
    if models.get('pipeline') is not None:
        return {"status": "already_initialized", "message": "Models already loaded"}
    
    try:
        await load_models()
        return {"status": "success", "message": "Models initialized successfully"}
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        return {"status": "error", "message": f"Initialization failed: {str(e)}"}


@app.get("/models/info")
async def model_info():
    """Get information about loaded models."""
    info = {
        "models_loaded": {
            key: model is not None 
            for key, model in models.items() 
            if key != 'startup_time'
        },
        "device": ModelConfig().device,
        "startup_time": models.get('startup_time', 0.0)
    }
    
    # Add model-specific info if available
    if models.get('deberta_trainer'):
        try:
            config_path = Path(ModelConfig().deberta_model_path) / "training_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    deberta_config = json.load(f)
                info['deberta_config'] = deberta_config
        except:
            pass
    
    return info

@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    total_requests = sum(len(requests) for requests in request_counts.values())
    active_clients = len([
        client for client, requests in request_counts.items()
        if any(time.time() - req_time < WINDOW_SIZE for req_time in requests)
    ])
    
    return {
        "total_requests": total_requests,
        "active_clients": active_clients,
        "rate_limit": f"{REQUEST_LIMIT} requests per {WINDOW_SIZE} seconds",
        "uptime": time.time() - models.get('startup_time', time.time())
    }

# Agent-specific endpoints
@app.post("/agents/detect")
async def agent_detect(request: TextInput):
    """Call DetectorAgent directly."""
    pipeline = models.get('pipeline')
    if pipeline is None or pipeline.detector_agent is None:
        raise HTTPException(status_code=503, detail="DetectorAgent not available")
    
    try:
        result = pipeline.detector_agent.execute({'text': request.text})
        return {
            "success": result.success,
            "data": result.data,
            "processing_time": result.processing_time,
            "agent_name": result.agent_name,
            "error": result.error_message
        }
    except Exception as e:
        logger.error(f"DetectorAgent failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DetectorAgent failed: {str(e)}")

@app.post("/agents/rationale")
async def agent_rationale(request: TextInput):
    """Call RationaleAgent directly."""
    pipeline = models.get('pipeline')
    if pipeline is None or pipeline.rationale_agent is None:
        raise HTTPException(status_code=503, detail="RationaleAgent not available")
    
    try:
        # First run detection
        detection_result = pipeline.detector_agent.execute({'text': request.text})
        if not detection_result.success:
            raise HTTPException(status_code=500, detail="Detection failed")
        
        # Then run rationale extraction
        input_data = {'text': request.text}
        input_data.update(detection_result.data)
        
        result = pipeline.rationale_agent.execute(input_data)
        return {
            "success": result.success,
            "data": result.data,
            "processing_time": result.processing_time,
            "agent_name": result.agent_name,
            "error": result.error_message
        }
    except Exception as e:
        logger.error(f"RationaleAgent failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RationaleAgent failed: {str(e)}")

@app.get("/docs-custom")
async def custom_docs():
    """Custom documentation with examples."""
    return {
        "title": "Agentic Hate Speech Detection & Counter-Speech API",
        "description": "This API provides endpoints for hate speech detection, rationale extraction, and counter-speech generation using advanced AI agents.",
        "examples": {
            "process": {
                "input": {
                    "text": "This group should not exist!",
                    "include_rationale": True,
                    "generate_counter_speech": True
                },
                "description": "Process text through complete pipeline"
            },
            "detect": {
                "input": {
                    "text": "This is a test message"
                },
                "description": "Run hate speech and sentiment detection only"
            },
            "batch": {
                "input": {
                    "texts": [
                        "First message to analyze",
                        "Second message to analyze"
                    ],
                    "include_rationale": True,
                    "generate_counter_speech": True
                },
                "description": "Process multiple texts in batch"
            }
        },
        "agents": {
            "DetectorAgent": "Performs multi-task hate speech and sentiment detection",
            "RationaleAgent": "Extracts explanations for detected hate speech",
            "CounterSpeechAgent": "Generates respectful counter-speech responses",
            "GuardrailAgent": "Validates generated responses for safety",
            "RetryAgent": "Handles regeneration with improved prompting",
            "DisplayAgent": "Formats results for frontend consumption"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": [
            "/health", "/process", "/detect", "/explain", "/generate", 
            "/validate", "/batch", "/models/info", "/stats", "/docs-custom"
        ]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Startup message
@app.on_event("startup")
async def startup_message():
    logger.info("Agentic Hate Speech Detection & Counter-Speech API started")
    logger.info("Available endpoints:")
    logger.info("  - GET  /health - Health check")
    logger.info("  - POST /process - Complete pipeline processing")
    logger.info("  - POST /detect - Detection only")
    logger.info("  - POST /explain - Detection with rationales")
    logger.info("  - POST /generate - Counter-speech generation")
    logger.info("  - POST /validate - Text validation")
    logger.info("  - POST /batch - Batch processing")
    logger.info("  - GET  /models/info - Model information")
    logger.info("  - GET  /stats - API statistics")

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    # Run server
    uvicorn.run(
        "deployment_api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info"
    )