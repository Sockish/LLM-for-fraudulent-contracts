import gradio as gr
import requests
import yaml
import os
import socket

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def get_local_ip():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def test_connection():
    """Test if the API server is running"""
    try:
        url = f"http://localhost:{config['api_port']}/health"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✅ Server Status: {data.get('status', 'unknown')} - {data.get('message', '')}"
        else:
            return f"❌ Server returned status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"❌ Connection error: {str(e)}"

def chat_with_memory(message, history):
    """Chat function that automatically handles history"""
    # FIXED: Properly manage conversation history
    if not history:
        # First message - start with system prompt
        api_history = [{"role": "system", "content": "You are a helpful legal assistant specialized in contract analysis and legal advice."}]
    else:
        # Subsequent messages - rebuild the full conversation history
        api_history = [{"role": "system", "content": "You are a helpful legal assistant specialized in contract analysis and legal advice."}]
        
        # Add all previous conversation turns
        for user_msg, assistant_msg in history:
            api_history.append({"role": "user", "content": user_msg})
            if assistant_msg:  # Only add if assistant response exists
                api_history.append({"role": "assistant", "content": assistant_msg})
    
    payload = {
        "message": message,
        "history": api_history  # Send the complete conversation history
    }
    
    url = f"http://localhost:{config['api_port']}/chat"
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection error: {str(e)}"

# Create combined interface
with gr.Blocks(title="LLM for Law", theme="soft") as demo:
    gr.Markdown("# ⚖️ LLM for Law")
    gr.Markdown("Your AI Legal Assistant for Professional Legal Guidance")
    
    # Display access information
    local_ip = get_local_ip()
    
    # Connection test section
    with gr.Row():
        with gr.Column(scale=3):
            status_output = gr.Textbox(
                label="🔧 API Server Status",
                value="Click 'Test Connection' to check server status",
                interactive=False
            )
        with gr.Column(scale=1):
            test_button = gr.Button("Test Connection", variant="secondary")
    
    # Chat interface
    chat_interface = gr.ChatInterface(
        fn=chat_with_memory,
        title="Chat with your AI Legal Assistant",
        description="Your AI specialist in law! Ask about contracts, legal principles, and more.",
        examples=[
            "Generate a fraudulent contract",
            "Analyze the contract and summarize the key points",
            "Help me fix the contract",
            "What are the main issues with the contract?",
            "What are the risks associated with this contract?",
        ],
        cache_examples=False,
    )
    
    # Connect the test function
    test_button.click(fn=test_connection, outputs=status_output)

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"⚖️  Starting LLM for Law interface...")
    print(f"🏠 Local access: http://localhost:{config['gradio_port']}")
    print(f"🌐 Network access: http://{local_ip}:{config['gradio_port']}")
    print(f"📡 API endpoint: http://{local_ip}:{config['api_port']}")
    
    demo.launch(
        server_port=config['gradio_port'],
        server_name="0.0.0.0",
        share=True,  # Should work after frpc download
        show_error=True,
        show_api=False
    )