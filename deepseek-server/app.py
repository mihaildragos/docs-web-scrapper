#!/usr/bin/env python3
"""
DeepSeek API Server - Simplified for Mac compatibility

A FastAPI server that provides a REST API for interacting with the DeepSeek LLM.
"""

import os
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/deepseek-llm-1.3b-base")
DEVICE = os.getenv("DEVICE", "cpu")  # Default to CPU for Mac
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "1024"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek API",
    description="REST API for interacting with the DeepSeek language model",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model state
model = None
tokenizer = None
generator = None
is_model_loaded = False
model_loading = False


class ModelLoadError(Exception):
    """Exception raised when model loading fails"""

    pass


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = TEMPERATURE
    do_sample: Optional[bool] = True
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0


class GenerateResponse(BaseModel):
    text: str
    finish_reason: str
    generated_tokens: int
    elapsed_time: float


async def load_model_async():
    """Load the model in background - simplified for macOS compatibility"""
    global model, tokenizer, generator, is_model_loaded, model_loading

    if is_model_loaded or model_loading:
        return

    model_loading = True
    try:
        logger.info(f"Loading model {MODEL_ID}...")
        start_time = time.time()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # Simple model loading for Mac compatibility
        logger.info(f"Loading model on {DEVICE}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map={"": DEVICE},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Create generator pipeline
        logger.info("Creating text generation pipeline")
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
        )

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f} seconds")
        is_model_loaded = True

    except Exception as e:
        model_loading = False
        logger.error(f"Failed to load model: {str(e)}")
        raise ModelLoadError(f"Failed to load model: {str(e)}")

    model_loading = False


@app.on_event("startup")
async def startup_event():
    """Start model loading when the API server starts"""
    background_tasks = BackgroundTasks()
    background_tasks.add_task(load_model_async)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "model": MODEL_ID, "loaded": is_model_loaded}


@app.get("/status")
async def status():
    """Get model status"""
    return {
        "model_id": MODEL_ID,
        "is_loaded": is_model_loaded,
        "is_loading": model_loading,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate text based on a prompt"""
    global model, tokenizer, generator, is_model_loaded

    if not is_model_loaded:
        if not model_loading:
            # Start loading if not already loading
            background_tasks.add_task(load_model_async)
        raise HTTPException(
            status_code=503, detail="Model is still loading. Please try again later."
        )

    try:
        start_time = time.time()

        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "do_sample": request.do_sample,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "repetition_penalty": request.repetition_penalty,
        }

        # Generate text
        outputs = generator(request.prompt, **gen_kwargs)

        # Extract generated text
        generated_text = outputs[0]["generated_text"]

        # Remove the prompt from the generated text
        if generated_text.startswith(request.prompt):
            generated_text = generated_text[len(request.prompt) :]

        elapsed_time = time.time() - start_time

        return GenerateResponse(
            text=generated_text,
            finish_reason="stop",
            generated_tokens=len(tokenizer.encode(generated_text)),
            elapsed_time=elapsed_time,
        )

    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
