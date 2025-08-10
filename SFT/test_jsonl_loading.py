#!/usr/bin/env python3

"""
Test script to validate JSONL loading and processing functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sft_utils import load_datasets, apply_chat_template, get_tokenizer
import json

def test_jsonl_loading():
    """Test loading JSONL data and processing it."""
    print("Testing JSONL loading...")
    
    # Test loading the dataset
    try:
        train_dataset, eval_dataset = load_datasets("parsed_contracts.jsonl")
        print(f"✓ Successfully loaded datasets")
        print(f"  - Train dataset size: {len(train_dataset)}")
        print(f"  - Eval dataset size: {len(eval_dataset)}")
        
        # Print first example
        print(f"\nFirst training example:")
        first_example = train_dataset[0]
        print(f"Input preview: {first_example['input'][:200]}...")
        print(f"Output preview: {first_example['output'][:200]}...")
        
    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return False
    
    return True

def test_chat_template():
    """Test applying chat template."""
    print("\nTesting chat template application...")
    
    try:
        # Create a dummy tokenizer for testing (you'd need to replace with actual model path)
        # For now, let's just test the template logic
        
        # Mock tokenizer with apply_chat_template method
        class MockTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                result = ""
                for message in messages:
                    if message['role'] == 'user':
                        result += f"### Problem: {message['content']}\n"
                    elif message['role'] == 'assistant':
                        result += f"### Solution: {message['content']}\n"
                if add_generation_prompt and messages[-1]['role'] == 'user':
                    result += "### Solution: "
                return result
        
        tokenizer = MockTokenizer()
        
        # Test with sample data
        example = {
            "input": "Sample contract text...",
            "output": "Sample analysis..."
        }
        
        result = apply_chat_template(example, tokenizer, "sft")
        print("✓ Successfully applied chat template")
        print(f"Formatted text preview: {result['text'][:300]}...")
        
    except Exception as e:
        print(f"✗ Failed to apply chat template: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Running JSONL loading tests...\n")
    
    success = True
    success &= test_jsonl_loading()
    success &= test_chat_template()
    
    print(f"\n{'All tests passed! ✓' if success else 'Some tests failed! ✗'}")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
