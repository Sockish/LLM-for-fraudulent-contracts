from gradio import Interface, Chatbot, Textbox, Dropdown, Button, State

def chat_with_api(message, history, persona):
    if history is None:
        history = []
    
    persona_prompts = {
        "Helpful": "You are a helpful assistant.",
        "Formal": "You are a very formal assistant.",
        "Casual": "You are a friendly, casual assistant.",
        "Scientist": "You are a knowledgeable scientist who explains concepts in detail."
    }
    system_prompt = {"role": "system", "content": persona_prompts.get(persona, "You are a helpful assistant.")}
    
    chat_history = [system_prompt]
    for user_msg, bot_msg in history:
        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": bot_msg})
    chat_history.append({"role": "user", "content": message})
    
    payload = {
        "message": message,
        "history": chat_history
    }
    url = "http://localhost:54223/chat"
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        bot_response = response.json().get("response", "")
    else:
        bot_response = "Error: " + response.text
        
    history = history + [(message, bot_response)]
    return history

def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Advanced LLM Chat Interface")
        gr.Markdown("Select a persona to influence the response style, type your message, and interact with the LLM API server.")
        
        with gr.Row():
            persona_dropdown = Dropdown(
                choices=["Helpful", "Formal", "Casual", "Scientist"],
                value="Helpful",
                label="Choose Persona"
            )
            clear_btn = Button("Clear Chat", variant="stop")
        
        chatbot = Chatbot(label="Chat Conversation")
        msg = Textbox(placeholder="Type your message here...", label="Your Message")
        submit_btn = Button("Send")
        
        state = State([])  # to store conversation history
        
        submit_btn.click(fn=chat_with_api, inputs=[msg, state, persona_dropdown], outputs=[chatbot, state])
        msg.submit(fn=chat_with_api, inputs=[msg, state, persona_dropdown], outputs=[chatbot, state])
        clear_btn.click(lambda: (None, []), None, [chatbot, state])
    
    return demo

if __name__ == "__main__":
    create_ui().launch()