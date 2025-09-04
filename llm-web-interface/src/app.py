import subprocess
import time
import sys
import yaml
import os
import requests
import signal

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def check_api_server():
    """Check if API server is running"""
    try:
        response = requests.get(f"http://localhost:{config['api_port']}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the API server"""
    print("🚀 Starting LLM API Server...")
    api_script = os.path.join(os.path.dirname(__file__), 'api', 'llm_server.py')
    process = subprocess.Popen([sys.executable, api_script])
    
    # Wait for server to start
    max_wait = 60  # seconds
    wait_time = 0
    while wait_time < max_wait:
        if check_api_server():
            print("✅ API Server is running!")
            return process
        time.sleep(2)
        wait_time += 2
        print(f"⏳ Waiting for API server... ({wait_time}s)")
    
    print("❌ API Server failed to start within 60 seconds")
    process.terminate()
    return None

def start_ui_server():
    """Start the UI server"""
    print("🎨 Starting Gradio UI Server...")
    ui_script = os.path.join(os.path.dirname(__file__), 'ui', 'gradio_app.py')
    process = subprocess.Popen([sys.executable, ui_script])
    return process

def main():
    processes = []
    
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down servers...")
        for p in processes:
            p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start API server
        api_process = start_api_server()
        if api_process:
            processes.append(api_process)
        else:
            print("Failed to start API server. Exiting.")
            return
        
        # Start UI server
        ui_process = start_ui_server()
        processes.append(ui_process)
        
        print(f"\n🎉 Servers are running!")
        print(f"📡 API Server: http://localhost:{config['api_port']}")
        print(f"🌐 UI Server: http://localhost:{config['gradio_port']}")
        print("Press Ctrl+C to stop all servers")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()