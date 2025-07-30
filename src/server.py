"""
MLX-optimized FastAPI server for the fine-tuned chatbot.
This server can load both PyTorch and MLX models.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from persona import get_persona

app = FastAPI(title="Tank GPT", description="AI Chatbot trained on your conversations")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global variables for model
model = None
tokenizer = None
model_type = None  # 'pytorch' or 'mlx'
current_persona = "default"  # Default persona to use


class ChatMessage(BaseModel):
    message: str
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.7
    persona: Optional[str] = None  # Allow persona override per message


class ChatResponse(BaseModel):
    response: str
    model_type: str


def load_mlx_model(model_path: str):
    """Load MLX model and tokenizer."""
    try:
        from mlx_lm import load, generate
        import mlx.core as mx
        
        print(f"Loading MLX model from {model_path}")
        print(f"MLX device: {mx.default_device()}")
        
        # Check for trained adapters
        adapter_path = Path(model_path) / "adapters"
        if adapter_path.exists():
            print(f"Loading model with LoRA adapters from {adapter_path}")
            # Load base model with adapters
            model, tokenizer = load("TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path=str(adapter_path))
        else:
            # Try loading as a complete model directory
            print(f"Loading complete model from {model_path}")
            model, tokenizer = load(model_path)
        
        return model, tokenizer, "mlx"
        
    except ImportError as e:
        print(f"MLX not available: {e}")
        return None, None, None
    except Exception as e:
        print(f"Error loading MLX model: {e}")
        return None, None, None


def load_pytorch_model(model_path: str):
    """Load PyTorch model with LoRA adapters."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        print(f"Loading PyTorch model from {model_path}")
        
        # Try to load as a LoRA model first
        try:
            base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Load LoRA adapters
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer, "pytorch"
            
        except Exception as lora_error:
            print(f"Failed to load as LoRA model: {lora_error}")
            
            # Try to load as a regular fine-tuned model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer, "pytorch"
            
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        return None, None, None


def load_chatbot_model():
    """Load the fine-tuned chatbot model (try MLX first, then PyTorch)."""
    global model, tokenizer, model_type
    
    # Try MLX model first (recommended)
    mlx_model_path = "assets/models/mlx-chat-model"
    if os.path.exists(mlx_model_path):
        print("Attempting to load MLX model...")
        model, tokenizer, model_type = load_mlx_model(mlx_model_path)
        if model is not None:
            print("Successfully loaded MLX model!")
            return True
    
    # Fall back to PyTorch model
    pytorch_model_path = "assets/models/chat-model"
    if os.path.exists(pytorch_model_path):
        print("Attempting to load PyTorch model...")
        model, tokenizer, model_type = load_pytorch_model(pytorch_model_path)
        if model is not None:
            print("Successfully loaded PyTorch model!")
            return True
    
    print("No trained model found. Please run fine-tuning first.")
    return False


def generate_response_mlx(prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
    """Generate response using MLX model."""
    from mlx_lm import generate
    
    try:
        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False
        )
        
        # Extract just the generated part (remove the prompt)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        print(f"Error generating MLX response: {e}")
        return "Sorry, I'm having trouble generating a response right now."


def generate_response_pytorch(prompt: str, max_length: int = 150, temperature: float = 0.7) -> str:
    """Generate response using PyTorch model."""
    import torch
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (remove the prompt)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        print(f"Error generating PyTorch response: {e}")
        return "Sorry, I'm having trouble generating a response right now."


@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    success = load_chatbot_model()
    if not success:
        print("Warning: No model loaded. Server will return error responses.")


@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface."""
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    
    with open(static_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Generate a chat response."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get the persona to use (from message or default)
    persona_name = message.persona or current_persona
    persona_text = get_persona(persona_name)
    
    # Create the conversation in the format the model was trained on
    # This matches the format from your training data
    conversation = [
        {"role": "system", "content": persona_text},
        {"role": "user", "content": message.message}
    ]
    
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback format
        prompt = f"<|system|>\n{persona_text}</s>\n<|user|>\n{message.message}</s>\n<|assistant|>\n"
    
    # Generate response based on model type
    if model_type == "mlx":
        response = generate_response_mlx(
            prompt, 
            max_tokens=message.max_length, 
            temperature=message.temperature
        )
    else:  # pytorch
        response = generate_response_pytorch(
            prompt,
            max_length=len(tokenizer.encode(prompt)) + message.max_length,
            temperature=message.temperature
        )
    
    return ChatResponse(response=response, model_type=model_type)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type or "none"
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Tank GPT server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "server_mlx:app" if __name__ == "__main__" else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
