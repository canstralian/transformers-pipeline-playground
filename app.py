import logging
from huggingface_hub import InferenceApi
from transformers import AutoTokenizer

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your API token and model repo ID
API_TOKEN = "hf_YourAPITokenHere"
MODEL_REPO = "Salesforce/codet5-base"  # adjust to the correct model repo

# Load tokenizer to check tokenization
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
except Exception as e:
    logger.exception("Error loading tokenizer:")
    raise

# Verify tokenization is correct
input_text = "def greet(user): print(f'hello <extra_id_0>!')"
try:
    tokenized = tokenizer(input_text, return_tensors="pt")
    logger.info(f"Token IDs: {tokenized['input_ids']}")
except Exception as e:
    logger.exception("Tokenization error:")
    raise

# Initialize remote inference API client
inference = InferenceApi(repo_id=MODEL_REPO, token=API_TOKEN)

def generate_code(prompt: str, max_length: int = 20):
    payload = {"inputs": prompt, "parameters": {"max_length": max_length}}
    try:
        response = inference(payload)
        # Check if response has an error key
        if isinstance(response, dict) and "error" in response:
            logger.error(f"Inference API error: {response['error']}")
            return None
        return response
    except Exception as e:
        logger.exception("Exception during remote inference:")
        return None

if __name__ == "__main__":
    result = generate_code(input_text)
    if result is not None:
        # Depending on the model's output format, adjust accordingly
        logger.info(f"Generated output: {result}")
    else:
        logger.error("Failed to generate output.")