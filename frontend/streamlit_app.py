#!/usr/bin/env python3
"""
Streamlit Frontend for Agentic Hate Speech Detection & Counter-Speech Generation System
Material Design themed interface with FastAPI backend integration
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.langchain_agents import AgenticPipeline
from backend.deberta_multitask import MultiTaskDeBERTa, MultiTaskTrainer
from backend.llama_counter_speech import LLaMA3CounterSpeechGenerator, CounterSpeechValidator

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional, Any

# Page configuration
st.set_page_config(
    page_title="Hate Speech Detection & Counter-Speech Generation System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Material Design CSS styling
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    :root {
        --primary: #6750A4;
        --on-primary: #FFFFFF;
        --primary-container: #EADDFF;
        --on-primary-container: #21005D;
        --secondary: #625B71;
        --surface: #FFFBFE;
        --on-surface: #1C1B1F;
        --error: #BA1A1A;
        --error-container: #FFDAD6;
    }
    
    .stApp {
        font-family: 'Roboto', sans-serif;
        background-color: var(--surface);
        color: var(--on-surface);
    }
    
    .main-header {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: var(--on-primary);
        box-shadow: 0 4px 12px rgba(103, 80, 164, 0.15);
    }
    
    .main-header h1 {
        margin: 0;
        font-weight: 500;
        font-size: 2.5rem;
    }
    
    .card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #E0E0E0;
    }
    
    .metric-card {
        background: var(--primary-container);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #E0E0E0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    .status-safe {
        background: #E8F5E8;
        color: #2E7D32;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
    }
    
    .status-unsafe {
        background: var(--error-container);
        color: var(--error);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
    }
    
    .stButton > button {
        background: var(--primary);
        color: var(--on-primary);
        border: none;
        border-radius: 20px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: 'Roboto', sans-serif;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--primary);
        opacity: 0.9;
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

class APIClient:
    """FastAPI client for backend communication."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    

    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else {"status": "error"}
        except:
            return {"status": "disconnected"}
    
    def process_text(self, text: str, include_rationale: bool = True, 
                    generate_counter_speech: bool = True) -> Dict[str, Any]:
        """Process text through complete pipeline."""
        try:
            payload = {
                "text": text,
                "include_rationale": include_rationale,
                "generate_counter_speech": generate_counter_speech
            }
            response = self.session.post(f"{self.base_url}/process", 
                                       json=payload, timeout=120)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_only(self, text: str) -> Dict[str, Any]:
        """Run detection only."""
        try:
            response = self.session.post(f"{self.base_url}/detect", 
                                       json={"text": text}, timeout=10)
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def batch_process(self, texts: List[str]) -> Dict[str, Any]:
        """Process multiple texts."""
        try:
            payload = {"texts": texts}
            response = self.session.post(f"{self.base_url}/batch", 
                                       json=payload, timeout=60)
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def initialize_session_state():
    """Initialize session state variables."""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient()
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

def render_header():
    """Render main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ AI Content Moderation</h1>
        <p>Advanced hate speech detection with empathetic counter-speech generation</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with controls and info."""
    with st.sidebar:
        st.markdown("### 🛠️ System Control")
        
        # API Configuration
        st.markdown("#### API Configuration")
        api_url = st.text_input("API URL", value="http://localhost:8000")
        
        if st.button("Update API URL"):
            st.session_state.api_client = APIClient(api_url)
            st.success("API URL updated!")
        
        # Health Check
        st.markdown("#### System Status")
        health = st.session_state.api_client.health_check()
        
        if health.get("status") == "healthy":
            st.success("🟢 System Healthy")
        elif health.get("status") == "degraded":
            st.warning("🟡 System Degraded")
        else:
            st.error("🔴 System Offline")
        
        # Model Status
        st.markdown("#### Model Loading Status")

        health = st.session_state.api_client.health_check()

        if health.get("status") == "disconnected":
            st.error("🔴 API Server Offline")
        elif health.get("models_loaded"):
            models = health["models_loaded"]
            
            # Check if any models are still loading
            loading_models = [name for name, loaded in models.items() if not loaded]
            
            if loading_models:
                st.warning("🟡 Models Loading...")
                st.info("⏳ Please wait for models to finish loading before analyzing text...")
                # Show loading progress for each model
                for model_name, is_loaded in models.items():
                    if not is_loaded:
                        st.markdown(f"**{model_name.title()}**: Loading...")
                    else:
                        st.success(f"✅ {model_name.title()}: Ready")
                
            


                    #     # Create a loading bar placeholder
                    #     if model_name == "llama":
                    #         st.markdown("Loading checkpoint shards...")
                    #         # Simulate the 4/4 progress you see in terminal
                    #         progress_placeholder = st.empty()
                    #         with progress_placeholder.container():
                    #             progress_bar = st.progress(0)
                    #             st.text("Initializing...")
                                
                    #     elif model_name == "deberta":
                    #         st.markdown("Loading DeBERTa model...")
                    #         progress_bar = st.progress(0)
                    #         st.text("Loading tokenizer and config...")
                            
                    #     else:
                    #         st.markdown(f"Loading {model_name}...")
                    #         progress_bar = st.progress(0)
                    # else:
                    #     st.success(f"✅ {model_name.title()} Ready")
            else:
                st.success("✅ All Models Loaded")
        else:
            st.info("🔄 Checking model status...")
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        # Processing Options
        st.markdown("#### Processing Options")
        include_rationale = st.checkbox("Include Rationale", value=True)
        generate_counter = st.checkbox("Generate Counter-Speech", value=True)
        
        # Store in session state
        st.session_state.include_rationale = include_rationale
        st.session_state.generate_counter = generate_counter
def update_loading_status():
    """Update loading status in real-time"""
    if 'last_health_check' not in st.session_state:
        st.session_state.last_health_check = 0
    
    current_time = time.time()
    
    # Check health every 2 seconds during loading
    if current_time - st.session_state.last_health_check > 2:
        st.session_state.last_health_check = current_time
        
        # This will trigger a rerun to update the sidebar
        health = st.session_state.api_client.health_check()
        
        # If models are still loading, auto-refresh
        if health.get("models_loaded"):
            loading_models = [name for name, loaded in health["models_loaded"].items() if not loaded]
            if loading_models:
                time.sleep(1)
                st.rerun()
def render_single_analysis():
    """Render single text analysis interface."""
    st.markdown("### 📝 Single Text Analysis")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste the text you want to analyze for hate speech...",
            height=100
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
        
        if text_input:
            st.info(f"Characters: {len(text_input)}/2000")
    
    # Processing and results
    if analyze_button and text_input:
        if len(text_input) > 2000:
            st.error("Text exceeds 2000 character limit!")
            return
        
        # Check if models are ready
        health = st.session_state.api_client.health_check()
        if health.get("status") != "healthy":
            st.error("⚠️ System not ready! Please wait for all models to finish loading.")
            st.info("Check the sidebar for model loading status.")
            return
        
        models_loaded = health.get("models_loaded", {})
        if not all(models_loaded.values()):
            st.error("⚠️ Models are still loading! Please wait before analyzing.")
            st.info("Check the sidebar for detailed loading status.")
            return
        # Show processing indicator
        with st.spinner("🤖 AI agents are analyzing your text..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress updates
            status_text.text("🔍 DetectorAgent: Analyzing content...")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            status_text.text("🧠 RationaleAgent: Extracting explanations...")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            if st.session_state.generate_counter:
                status_text.text("💬 CounterSpeechAgent: Generating response...")
                progress_bar.progress(60)
                time.sleep(0.5)
                
                status_text.text("🛡️ GuardrailAgent: Validating safety...")
                progress_bar.progress(80)
                time.sleep(0.5)
            
            status_text.text("📊 DisplayAgent: Formatting results...")
            progress_bar.progress(90)
            
            # Process the text
            result = st.session_state.api_client.process_text(
                text_input,
                st.session_state.include_rationale,
                st.session_state.generate_counter
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(1)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
        
        # Display results
        if result.get("success"):
            render_analysis_results(result, text_input)
            
            # Save to history
            result['timestamp'] = datetime.now()
            st.session_state.processing_history.append(result)
        else:
            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

def render_analysis_results(result: Dict[str, Any], original_text: str):
    """Render analysis results with Material Design styling."""
    with st.expander("🐛 Debug Information"):
        st.json(result)
    # Detection Results Card
    st.markdown("### 🎯 Detection Results")
    
    col1, col2, col3 = st.columns(3)
    
    detection = result.get('detection', {})
    hate_score = detection.get('hate_score', 0)
    sentiment = detection.get('sentiment', 'unknown')
    confidence = detection.get('confidence', 0)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{hate_score:.2%}</div>
            <div class="metric-label">Hate Speech Probability</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sentiment_color = {
            'positive': '#4CAF50',
            'negative': '#F44336',
            'neutral': '#FF9800'
        }.get(sentiment.lower(), '#9E9E9E')
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {sentiment_color};">{sentiment.title()}</div>
            <div class="metric-label">Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{confidence:.2%}</div>
            <div class="metric-label">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk Level Indicator
    if hate_score > 0.7:
        risk_level = "HIGH RISK"
        st.markdown(f'<div class="status-unsafe">⚠️ {risk_level}</div>', unsafe_allow_html=True)
    elif hate_score > 0.3:
        risk_level = "MODERATE RISK"
        st.markdown(f'<div class="status-unsafe">⚡ {risk_level}</div>', unsafe_allow_html=True)
    else:
        risk_level = "LOW RISK"
        st.markdown(f'<div class="status-safe">✅ {risk_level}</div>', unsafe_allow_html=True)
    
    # Rationale Section
    if st.session_state.include_rationale and 'rationale' in result:
        st.markdown("### 🔍 Explanation & Rationale")
        
        rationale = result['rationale']
        
        # Token-level rationale
        if rationale.get('token_level'):
            st.markdown("**Key Concerning Terms:**")
            tokens = rationale['token_level']
            token_html = " ".join([f'<span class="status-unsafe">{token}</span>' for token in tokens])
            st.markdown(token_html, unsafe_allow_html=True)
        
        # Sentence-level explanation
        if rationale.get('sentence_level'):
            st.markdown("**Analysis Summary:**")
            st.info(rationale['sentence_level'])
        
        # Detailed explanation
        if rationale.get('explanation'):
            st.markdown("**Detailed Explanation:**")
            st.markdown(rationale['explanation'])
    
    # Counter-Speech Section
    if st.session_state.generate_counter and 'counter_speech' in result:
        st.markdown("### 💬 Generated Counter-Speech")
        
        counter_speech = result['counter_speech']
        validation = result.get('validation', {})
        
        if validation.get('is_safe', False):
            st.success("✅ Generated response passed safety validation")
        else:
            st.warning("⚠️ Generated response required multiple attempts")
        
        # Display the counter-speech
        st.markdown(f"""
        <div class="card">
            <h4>💬 Empathetic Response</h4>
            <p style="font-size: 1.1rem; line-height: 1.6;">{counter_speech}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Validation metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Response Safety", 
                     "✅ Safe" if validation.get('is_safe') else "❌ Unsafe")
        
        with col2:
            st.metric("Response Hate Score", 
                     f"{validation.get('hate_score', 0):.2%}")
        
        with col3:
            st.metric("Generation Attempts", 
                     validation.get('attempts', 0))
    
    # Processing Metadata
    with st.expander("📊 Processing Details"):
        meta = result.get('meta', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Processing Time", f"{meta.get('processing_time', 0):.2f}s")
            st.metric("Agent Calls", meta.get('agent_calls', 0))
        
        with col2:
            st.metric("Retry Attempts", meta.get('retries', 0))
            
            model_versions = meta.get('model_versions', {})
            if model_versions:
                st.markdown("**Model Versions:**")
                for model, version in model_versions.items():
                    st.text(f"• {model}: {version}")

def render_batch_analysis():
    """Render batch analysis interface."""
    st.markdown("### 📊 Batch Analysis")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"],
        horizontal=True
    )
    
    texts_to_analyze = []
    
    if input_method == "Text Input":
        # Manual text input
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            placeholder="Enter each text on a new line...\nUp to 50 texts supported",
            height=150
        )
        
        if batch_text:
            texts_to_analyze = [line.strip() for line in batch_text.split('\n') 
                              if line.strip()]
            st.info(f"Found {len(texts_to_analyze)} texts to analyze")
    
    else:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a text file or CSV:",
            type=['txt', 'csv']
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'text' in df.columns:
                        texts_to_analyze = df['text'].dropna().tolist()
                    else:
                        st.error("CSV must have a 'text' column")
                else:
                    content = uploaded_file.read().decode('utf-8')
                    texts_to_analyze = [line.strip() for line in content.split('\n') 
                                      if line.strip()]
                
                st.info(f"Loaded {len(texts_to_analyze)} texts from file")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Processing controls
    if texts_to_analyze:
        if len(texts_to_analyze) > 50:
            st.warning("⚠️ Too many texts! Only first 50 will be processed.")
            texts_to_analyze = texts_to_analyze[:50]
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🚀 Process Batch", type="primary"):
                process_batch(texts_to_analyze)
        
        with col2:
            st.info(f"Ready to process {len(texts_to_analyze)} texts")

def process_batch(texts: List[str]):
    """Process batch of texts."""
    with st.spinner(f"Processing {len(texts)} texts..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process batch
        result = st.session_state.api_client.batch_process(texts)
        
        if 'error' not in result:
            st.session_state.batch_results = result
            
            # Show summary
            total = result.get('total_processed', 0)
            success = result.get('success_count', 0)
            processing_time = result.get('total_processing_time', 0)
            
            progress_bar.progress(100)
            status_text.success(f"✅ Processed {success}/{total} texts in {processing_time:.1f}s")
            
            # Display results summary
            render_batch_results(result)
        else:
            st.error(f"Batch processing failed: {result['error']}")

def render_batch_results(batch_result: Dict[str, Any]):
    """Render batch processing results."""
    st.markdown("### 📈 Batch Results")
    
    results = batch_result.get('results', [])
    if not results:
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_processed = batch_result.get('total_processed', 0)
    success_count = batch_result.get('success_count', 0)
    total_time = batch_result.get('total_processing_time', 0)
    avg_time = total_time / total_processed if total_processed > 0 else 0
    
    with col1:
        st.metric("Total Processed", total_processed)
    
    with col2:
        st.metric("Success Rate", f"{success_count}/{total_processed}")
    
    with col3:
        st.metric("Total Time", f"{total_time:.1f}s")
    
    with col4:
        st.metric("Avg Time/Text", f"{avg_time:.2f}s")
    
    # Create results dataframe
    results_data = []
    for i, result in enumerate(results):
        if result.get('success'):
            detection = result.get('detection', {})
            validation = result.get('validation', {})
            meta = result.get('meta', {})
            
            results_data.append({
                'ID': i + 1,
                'Text Preview': result.get('original', '')[:50] + '...',
                'Hate Score': f"{detection.get('hate_score', 0):.2%}",
                'Sentiment': detection.get('sentiment', 'unknown').title(),
                'Safe Response': '✅' if validation.get('is_safe', False) else '❌',
                'Processing Time': f"{meta.get('processing_time', 0):.2f}s"
            })
        else:
            results_data.append({
                'ID': i + 1,
                'Text Preview': result.get('original', '')[:50] + '...',
                'Hate Score': 'Error',
                'Sentiment': 'Error',
                'Safe Response': '❌',
                'Processing Time': 'N/A'
            })
    
    if results_data:
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def render_history():
    """Render processing history."""
    st.markdown("### 📚 Processing History")
    
    if not st.session_state.processing_history:
        st.info("No processing history yet. Analyze some texts to see results here!")
        return
    
    # History controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.processing_history = []
            st.success("History cleared!")
            st.rerun()
    
    with col2:
        export_history = st.button("📤 Export History")
    
    # Display history
    for i, result in enumerate(reversed(st.session_state.processing_history)):
        timestamp = result.get('timestamp', datetime.now())
        original_text = result.get('original', 'Unknown text')
        
        with st.expander(f"Analysis {len(st.session_state.processing_history) - i} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Original Text:**")
                st.text(original_text[:200] + '...' if len(original_text) > 200 else original_text)
            
            with col2:
                detection = result.get('detection', {})
                hate_score = detection.get('hate_score', 0)
                sentiment = detection.get('sentiment', 'unknown')
                
                st.metric("Hate Score", f"{hate_score:.2%}")
                st.metric("Sentiment", sentiment.title())
            
            # Counter-speech if available
            if result.get('counter_speech'):
                st.markdown("**Generated Counter-Speech:**")
                st.info(result['counter_speech'])
def initialize_system():
    """Initialize the system by loading models."""
    with st.spinner("🤖 Initializing AI models..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Check if system is already initialized
            status_text.text("🔍 Checking API server...")
            progress_bar.progress(10)

            health = st.session_state.api_client.health_check()
            if health.get("status") == "disconnected":
                st.error("❌ API server is not running! Please start the FastAPI server first.")
                st.code("python deployment_api.py", language="bash")
                return
            
            progress_bar.progress(25)

            if health.get("status") == "healthy":
                st.success("✅ System already initialized!")
                return
            
            status_text.text("🔄 Starting model initialization...")
            progress_bar.progress(25)
            time.sleep(3)
            
            progress_bar.progress(75)
            status_text.text("🧠 Checking model status...")

            # Call initialization endpoint (you'll need to add this to your FastAPI)
            response = st.session_state.api_client.session.post(
                f"{st.session_state.api_client.base_url}/initialize",
                timeout=120
            )
            
            progress_bar.progress(75)
            status_text.text("🧠 Loading models...")
            time.sleep(2)
            
            progress_bar.progress(100)
            status_text.text("✅ System initialized successfully!")
            
            if response.status_code == 200:
                st.success("🚀 System initialized and ready!")
            else:
                st.error("❌ Initialization failed!")
                
        except Exception as e:
            st.error(f"❌ Initialization error: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def main():
    """Main application function."""
    # Initialize
    load_css()
    initialize_session_state()

    # Update loading status
    update_loading_status()
    
    # Header
    render_header()
    
    # Sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📝 Single Analysis", 
        "📊 Batch Analysis", 
        "📚 History"
    ])
    
    with tab1:
        render_single_analysis()
    
    with tab2:
        render_batch_analysis()
    
    with tab3:
        render_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "🛡️ AI Content Moderation System | "
        "Powered by DeBERTa + LLaMA 3 | "
        "Built with ❤️ using Streamlit & FastAPI"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()