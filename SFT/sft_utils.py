# coding=utf-8
# Copyright 2024 The Numina Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datasets
import dataclasses
import logging
import os
import subprocess
import sys
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, NewType, Optional, Tuple, Union

from transformers import AutoTokenizer,  PreTrainedTokenizer


CHAT_TEMPLATE = "{% for message in messages %}{% if (message['role'] == 'system')%}{{ '' }}{% elif (message['role'] == 'user')%}{{ '### Problem: ' + message['content'] + '\n' }}{% elif (message['role'] == 'assistant')%}{{ '### Solution: ' + message['content'] + '\n' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ '### Solution: ' }}{% endif %}{% endfor %}"


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation"],
):
    if task in ["sft", "generation"]:
        # Handle JSONL format with "input" and "output" fields
        if "input" in example and "output" in example:
            messages = [
                {"role": "system", "content": ""},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]}
            ]
        else:
            messages = example["messages"]
            # We add an empty system message if there is none
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})
        
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation']}"
        )
    return example



def get_tokenizer(model_name_or_path, set_pad_token: bool = True) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision="main",
        trust_remote_code=False,
    )

    if set_pad_token is True and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    tokenizer.chat_template = CHAT_TEMPLATE

    return tokenizer


def load_jsonl_dataset(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_train_eval_split(data, eval_ratio=0.1):
    """Split data into train and eval sets."""
    import random
    random.shuffle(data)
    split_idx = int(len(data) * (1 - eval_ratio))
    return data[:split_idx], data[split_idx:]


def load_datasets(dataset_name_or_path):
    """Load datasets from disk or JSONL file."""
    if dataset_name_or_path.endswith('.jsonl'):
        # Load JSONL file
        data = load_jsonl_dataset(dataset_name_or_path)
        train_data, eval_data = create_train_eval_split(data)
        
        train_dataset = datasets.Dataset.from_list(train_data)
        eval_dataset = datasets.Dataset.from_list(eval_data)
    else:
        # Load from disk (original functionality)
        raw_datasets = datasets.load_from_disk(dataset_name_or_path)
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.shard(num_shards=10, index=0)
        eval_dataset = raw_datasets["test"]
    
    return train_dataset, eval_dataset


def tokenize_and_format_data(examples, tokenizer, max_length=2048):
    """
    Tokenize and format data with proper padding and truncation.
    Handles varying input and output lengths.
    """
    formatted_texts = []
    
    for example in examples["text"]:
        # Tokenize the text
        tokenized = tokenizer(
            example,
            truncation=True,
            max_length=max_length,
            padding=False,  # We'll pad in the collator
            return_tensors=None,
        )
        formatted_texts.append(tokenized)
    
    # Collect all tokenized data
    input_ids = [item["input_ids"] for item in formatted_texts]
    attention_mask = [item["attention_mask"] for item in formatted_texts]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids.copy(),  # For causal LM, labels are the same as input_ids
    }


def create_data_collator(tokenizer, max_length=2048):
    """
    Create a data collator that handles padding for varying sequence lengths.
    """
    def collate_fn(batch):
        # Handle both dict and direct access formats
        if isinstance(batch[0], dict) and "text" in batch[0]:
            # Extract text and tokenize on the fly
            texts = [item["text"] for item in batch]
            tokenized_batch = []
            
            for text in texts:
                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None,
                )
                tokenized_batch.append(tokenized)
            
            input_ids = [item["input_ids"] for item in tokenized_batch]
            attention_masks = [item["attention_mask"] for item in tokenized_batch]
        else:
            # Already tokenized
            input_ids = [item["input_ids"] for item in batch]
            attention_masks = [item["attention_mask"] for item in batch]
        
        # Find max length in batch
        max_len = min(max([len(ids) for ids in input_ids]), max_length)
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []
        
        for ids, mask in zip(input_ids, attention_masks):
            # Truncate if necessary
            if len(ids) > max_len:
                ids = ids[:max_len]
                mask = mask[:max_len]
            
            # Pad to max_len
            padding_length = max_len - len(ids)
            padded_ids = ids + [tokenizer.pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length
            
            # For labels, we set padded tokens to -100 (ignored in loss calculation)
            labels = padded_ids.copy()
            for i in range(len(ids), max_len):
                labels[i] = -100
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
            padded_labels.append(labels)
        
        return {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(padded_attention_masks),
            "labels": torch.tensor(padded_labels),
        }
    
    return collate_fn

