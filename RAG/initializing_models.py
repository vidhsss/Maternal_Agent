"""
initializing_models.py
---------------------
Provides model initialization utilities for various HuggingFace models, including quantization options for efficient loading.

Functions:
    - get_mixtral_model: Loads Mixtral-8x7B-Instruct in 8-bit mode.
    - get_llama_model: Loads Llama-3.2-3B-Instruct on CUDA.
    - get_falcon_model: Loads Falcon-7B on CUDA.
    - get_medalpaca_model: Loads Medalpaca-7B on CUDA.

Note: For use in other modules, import and call the relevant function.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Example function for loading Mixtral

def get_mixtral_model():
    """Load Mixtral-8x7B-Instruct in 8-bit mode."""
    model_name = "/data/models/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ Mixtral loaded successfully in 8-bit mode.")
    return model, tokenizer

def get_llama_model():
    """Load Llama-3.2-3B-Instruct on CUDA."""
    model_name =  "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    return model, tokenizer

def get_falcon_model():
    """Load Falcon-7B on CUDA."""
    model_name =  "/data/models/huggingface/tiiuae/falcon-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    return model, tokenizer

def get_medalpaca_model():
    """Load Medalpaca-7B on CUDA."""
    model_name = "medalpaca/medalpaca-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    return model, tokenizer

model_name = "/data/models/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1"

# Enable 8-bit quantization to reduce VRAM usage
bnb_config = BitsAndBytesConfig(load_in_8bit=True)  # Use load_in_4bit=True if needed

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in optimized mode
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("✅ Mixtral loaded successfully in 8-bit mode.")

model_name =  "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to("cuda")

model_name =  "/data/models/huggingface/tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to("cuda")


tokenizer = AutoTokenizer.from_pretrained("medalpaca/medalpaca-7b", legacy=True)
model = AutoModelForCausalLM.from_pretrained(
            "medalpaca/medalpaca-7b"
        ).to("cuda")