from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

app = FastAPI()

# Global variables to store model and tokenizer
model = None
tokenizer = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    updated_history: List[ChatMessage]

def load_model(model_path: str):
    """Load the model and tokenizer"""
    global model, tokenizer
    try:
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda:0",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def chat_resp(user_prompt=None, history=[]):
    """Generate response using the loaded model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    generation_args = {
        "max_new_tokens": config['max_tokens'],
        "return_full_text": False,
        "temperature": config['temperature'],
        "do_sample": True,
    }
    
    # Build messages list - properly handle history
    messages = []
    
    # FIXED: Always ensure we have a system message at the start
    if history:
        # Use the existing history (which should already contain system message)
        for msg in history:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                messages.append(msg)
    else:
        # Only add system message if no history exists
        messages.append({"role": "system", "content": "You are a helpful legal assistant specialized in contract analysis and legal advice."})
    
    # Add current user message
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    
    try:
        print(f"Sending {len(messages)} messages to model")  # Debug info
        print(f"Message history: {[msg['role'] for msg in messages]}")  # Debug roles
        output = pipe(messages, **generation_args)
        response_text = output[0].get("generated_text", "") if output else ""
        
        # Return the response and updated history
        updated_messages = messages + [{"role": "assistant", "content": response_text}]
        return response_text, updated_messages
        
    except Exception as e:
        print(f"Error generating response: {e}")
        error_msg = f"Error generating response: {str(e)}"
        updated_messages = messages + [{"role": "assistant", "content": error_msg}]
        return error_msg, updated_messages

@app.post("/chat", response_model=ChatResponse)
async def run_chat(request: ChatRequest):
    """Handle chat requests with history management"""
    try:
        response_text, updated_history = chat_resp(
            user_prompt=request.message, 
            history=request.history
        )
        
        # Convert to ChatMessage objects for response
        history_messages = [
            ChatMessage(role=msg["role"], content=msg["content"]) 
            for msg in updated_history
        ]
        
        return ChatResponse(
            response=response_text,
            updated_history=history_messages
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM API server!"}

@app.get("/health")
async def health_check():
    """Check if model is loaded and ready"""
    if model is None or tokenizer is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "message": "Model is ready"}

if __name__ == "__main__":
    # Load model from config
    model_path = config['model_path']
    load_model(model_path)
    
    # Start the server using config port
    uvicorn.run(app, host="0.0.0.0", port=config['api_port'])