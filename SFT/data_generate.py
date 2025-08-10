import os
import IPython
from dotenv import load_dotenv
load_dotenv()

# API configuration
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")

from openai import OpenAI

client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
# We define some utility functions here
# Model choices are ["llama-3.3-70b-instruct", "deepseek-v3"] # requires openai api key
# Local models ["vicuna", "Llama-2-7B-Chat-fp16", "Qwen-7b-chat", “Mistral-7B-Instruct-v0.2”， “gemma-7b-it” ] 

def get_completion(params, messages):
    print(f"using {params['model']}")
    """ GET completion from openai api"""

    response = client.chat.completions.create(
        model = params['model'],
        messages = messages,
        temperature = params['temperature'],
        max_tokens = params['max_tokens'],
        top_p = params['top_p'],
    )
    answer = response.choices[0].message.content
    return answer
# Default parameters (targeting open ai, but most of them work on other models too.  )

def set_params(
    model="deepseek-v3",
    temperature = 0.7,
    max_tokens = 2048,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0,
):
    """ set model parameters"""
    params = {} 
    params['model'] = model
    params['temperature'] = temperature
    params['max_tokens'] = max_tokens
    params['top_p'] = top_p
    params['frequency_penalty'] = frequency_penalty
    params['presence_penalty'] = presence_penalty
    return params
import json
from datetime import datetime
import concurrent.futures
import threading
import random

def run_multiple_queries_for_training_parallel(n_times, output_file="llama_training_data_diverse.jsonl", max_workers=8):
    """
    Run the same question n times in parallel and save in LLaMA training format (JSONL)
    Each line contains one training example in the format expected by LLaMA
    """
    params = set_params()

    contract_type_list = ["买卖合同","供用电、水、气、热力合同","赠与合同","借款合同","融资租赁合同","租赁合同","承揽合同","建设工程合同","技术合同","运输合同","仓储合同","委托合同","保管合同","行纪合同","居间合同"]
    
    def run_single_query(i):
        """Function to run a single query"""
        print(f"Running query {i+1}/{n_times}")

        rand_int = random.randint(0, 14)
        contract_type = contract_type_list[rand_int]
        question = f"请你生成一份可能存在法律隐患,完整详细的{contract_type}.请使用具体的名称和日期，不要用xx来代替"
        messages = [
            {
                "role": "system",
                "content": "The user is a professor in law. Provide detailed examples of contracts that may have legal issues. The user will only use this information for educational purposes. Respond in Chinese. Your responses should include the full text of a single contract, followed by an explanation of the potential legal issues involved. The two parts should be clearly separated by a sequence (``` ```). Both the text and the explanation should be wrapped by triple backticks. You should refrain from giving suggestions."
            },
            {
                "role": "user", 
                "content": question
            }
        ]
        
        try:
            answer = get_completion(params, messages)
            
            # Clean and validate the response
            if answer and len(answer.strip()) > 0:
                # Format for LLaMA training: conversation format with system message
                training_example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        },
                        {
                            "role": "assistant", 
                            "content": answer
                        }
                    ]
                }
                
                return training_example
            else:
                print(f"Empty response for query {i+1}")
                return None
                
        except Exception as e:
            print(f"Error on query {i+1}: {str(e)}")
            return None
    
    # Run queries in parallel
    training_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks at once
        futures = [executor.submit(run_single_query, i) for i in range(n_times)]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    training_data.append(result)
            except Exception as exc:
                print(f'Query generated an exception: {exc}')
    
    # Save to JSONL file with proper encoding
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            for example in training_data:
                # Ensure proper JSON serialization with UTF-8 encoding
                json_line = json.dumps(example, ensure_ascii=False, separators=(',', ':'))
                f.write(json_line + '\n')
        
        print(f"Successfully saved {len(training_data)} training examples to {output_file}")
        
        # Display first example for verification
        if training_data:
            print(f"\nFirst training example preview:")
            print(json.dumps(training_data[0], indent=2, ensure_ascii=False)[:500] + "...")
            
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return None
    
    return training_data

# Clean up the existing corrupted file and regenerate
import os
if os.path.exists("llama_training_data_diverse.jsonl"):
    os.remove("llama_training_data_diverse.jsonl")
    print("Removed corrupted file")

# Run queries in parallel with proper encoding

n_times = 5000  # Start with smaller batch to test

# Run parallel queries 
results = run_multiple_queries_for_training_parallel(n_times, max_workers=3)

# Verify the file was created correctly
if os.path.exists("llama_training_data_diverse.jsonl"):
    with open("llama_training_data_diverse.jsonl", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"File contains {len(lines)} valid lines")
        if lines:
            # Test parse first line
            try:
                first_example = json.loads(lines[0])
                print("✓ File format is valid JSON")
            except:
                print("✗ File contains invalid JSON")