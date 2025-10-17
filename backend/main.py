from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import subprocess
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("predict.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Global variables to cache the model
_model = None
_tokenizer = None
_model_loaded = False

class Product(BaseModel):
    name: str
    description: str

def load_model():
    """Load model once and cache it"""
    global _model, _tokenizer, _model_loaded
    
    if _model_loaded:
        logging.info("Model already loaded, using cached version.")
        return _model, _tokenizer
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'product-price-predictor')
    base_model_name = "distilgpt2"
    
    logging.info("Loading model...")
    
    # Load base model
    logging.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Load PEFT adapter with error handling
    logging.info(f"Loading PEFT adapter from: {model_path}")
    try:
        _model = PeftModel.from_pretrained(base_model, model_path)
        _model.eval()
    except ValueError as e:
        if "Target modules" in str(e):
            logging.error(f"Model incompatibility detected: {e}")
            logging.error("The saved model was trained with different target modules.")
            logging.error("Please delete the model folder and retrain:")
            logging.error(f"Folder to delete: {model_path}")
            raise ValueError(
                "Model incompatibility detected. The model needs to be retrained. "
                f"Please delete the folder '{model_path}' and call the /train endpoint again."
            )
        raise
    
    logging.info("Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Add pad token if it doesn't exist
    if _tokenizer.pad_token is None:
        logging.info("Adding pad token to tokenizer.")
        _tokenizer.pad_token = _tokenizer.eos_token
    
    _model_loaded = True
    logging.info("Model loaded successfully!")
    
    return _model, _tokenizer

def run_training():
    script_path = os.path.join(os.path.dirname(__file__), '..', 'llm', 'qlora_fine_tuning.py')
    subprocess.run(['python', script_path])
    
    # Reset model cache after training
    global _model_loaded
    _model_loaded = False

@app.get("/")
async def root():
    return {"message": "Product Price Prediction API is running"}

@app.post("/train")
async def train(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_training)
    return {"message": "Training started in the background."}

@app.post("/predict")
async def predict(product: Product):
    logging.info("Received prediction request.")
    logging.info(f"Product Name: {product.name}")
    logging.info(f"Product Description: {product.description}")

    model_path = os.path.join(os.path.dirname(__file__), '..', 'product-price-predictor')
   
    # Check if model exists
    if not os.path.exists(model_path):
        logging.error("Model not found.")
        return {"error": "Model not found. Please train the model first."}
    
    try:
        # Load model (cached)
        model, tokenizer = load_model()
        
        # Create the prompt matching training format
        prompt = f"""Product: {product.name}
Description: {product.description}
Price: $"""
        
        logging.info(f"Generated prompt:\n{prompt}")
       
        # Tokenize the prompt
        logging.info("Tokenizing prompt...")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
        
        # Move inputs to same device as model
        if torch.cuda.is_available():
            logging.info("Moving inputs to CUDA device.")
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate the prediction
        logging.info("Generating prediction...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=10,
                min_new_tokens=1,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the generated text
        logging.info("Decoding generated text...")
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Full generated text from model: {predicted_text}")
        
        # Extract only the new generated part
        generated_part = predicted_text[len(prompt):].strip()
        logging.info(f"Generated part (model response): {generated_part}")
        
        # Extract price from the generated text
        price = extract_price(generated_part)
        
        if price:
            response = {
                "product": product.name, 
                "predicted_price": float(price),
                "raw_output": generated_part
            }
            logging.info(f"Sending response: {response}")
            return response
        else:
            response = {
                "error": "Could not extract a valid price from the model's response.",
                "model_response": generated_part,
                "suggestion": "The model needs more training data. Current dataset might be too small (needs 100+ samples)."
            }
            logging.warning(f"Could not extract price. Sending error response: {response}")
            return response
            
    except Exception as e:
        import traceback
        logging.error(f"Error during prediction: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "error": f"Error during prediction: {str(e)}",
            "traceback": traceback.format_exc()
        }

def extract_price(text):
    """Improved price extraction with better pattern matching"""
    logging.info(f"Extracting price from: '{text}'")
    
    # Clean the text first
    text = text.strip()
    
    # Pattern 1: Look for dollar amounts with decimals (most reliable)
    dollar_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*\.\d{2})'
    matches = re.findall(dollar_pattern, text)
    
    if matches:
        clean_price = matches[0].replace(',', '')
        value = float(clean_price)
        logging.info(f"Valid price found via decimal pattern: {value}")
        return str(value)
    
    # Pattern 2: Look for dollar amounts without decimals
    dollar_pattern_simple = r'\$?\s*(\d{1,5})'
    matches = re.findall(dollar_pattern_simple, text)
    
    # Filter reasonable prices and take the most likely one
    reasonable_prices = []
    for match in matches:
        try:
            value = int(match)
            # Reasonable price range for consumer products
            if 50 <= value <= 5000:
                reasonable_prices.append(value)
        except ValueError:
            continue
    
    if reasonable_prices:
        # Take the smallest reasonable price (often the correct one)
        price = min(reasonable_prices)
        logging.info(f"Valid price found via integer pattern: {price}")
        return str(price)
    
    # Pattern 3: Extract numbers and filter
    all_numbers = re.findall(r'\d+', text)
    for num in all_numbers:
        try:
            value = int(num)
            if 50 <= value <= 5000:
                logging.info(f"Valid price found via fallback: {value}")
                return str(value)
        except ValueError:
            continue
    
    logging.warning(f"No valid price found in text: '{text}'")
    return None