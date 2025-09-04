def load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    from transformers import pipeline

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = pipe(prompt, max_length=150, num_return_sequences=1)
    return response[0]['generated_text'] if response else ""