#!/usr/bin/env python3
"""
Model Download Script - Simplified for Mac compatibility

This script pre-downloads the DeepSeek model to avoid downloading it at runtime.
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Get the model ID from environment or use default
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/deepseek-llm-1.3b-base")
print(f"Downloading model: {MODEL_ID}")

# Download the tokenizer
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained("./model_cache")

# Download the model with appropriate configuration for Mac
print("Downloading model for CPU...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": "cpu"},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# Save the model
model.save_pretrained("./model_cache")
print("Model downloaded successfully!")
