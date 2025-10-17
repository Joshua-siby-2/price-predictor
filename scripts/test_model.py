"""
Test script to verify model training and predictions
Run this after training to check if the model works correctly
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def test_model():
    """Test the trained model with sample inputs"""
    
    model_path = "models/product-price-predictor"
    base_model_name = "gpt2"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ ERROR: Model not found!")
        print(f"   Please train the model first. Looking for: {model_path}")
        return
    
    print("🔄 Loading model...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Load PEFT adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!\n")
    
    # Test cases based on training data patterns
    test_cases = [
        {
            "name": "Laptop",
            "description": "Laptop with 8GB RAM, 256GB SSD, Intel i5 processor",
            "expected": "~2600"
        },
        {
            "name": "Laptop",
            "description": "Laptop with 16GB RAM, 512GB SSD, Intel i5 processor",
            "expected": "~2727"
        },
        {
            "name": "Laptop",
            "description": "Laptop with 8GB RAM, 256GB SSD, AMD Ryzen 5 processor",
            "expected": "~2200"
        },
        {
            "name": "Smartphone",
            "description": "Smartphone with 6GB RAM, 128GB storage",
            "expected": "~300"
        },
        {
            "name": "Smartphone",
            "description": "Smartphone with 16GB RAM, 512GB storage",
            "expected": "~892"
        },
    ]
    
    print("=" * 80)
    print("TESTING MODEL PREDICTIONS")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}:")
        print(f"   Product: {test['name']}")
        print(f"   Description: {test['description']}")
        print(f"   Expected Price: {test['expected']}")
        
        # Create prompt (matching training format)
        prompt = f"""Product: {test['name']}
Description: {test['description']}
Price: """
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with deterministic settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                min_new_tokens=1,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
            )
        
        # Decode
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part
        generated_part = predicted_text[len(prompt):].strip()
        
        # Clean output (remove non-ASCII)
        generated_part = generated_part.encode('ascii', 'ignore').decode('ascii').strip()
        
        print(f"   Model Output: '{generated_part}'")
        
        # Try to extract number
        import re
        numbers = re.findall(r'\d+', generated_part)
        if numbers:
            price = numbers[0]
            print(f"   ✅ Extracted Price: {price}")
        else:
            print(f"   ❌ Could not extract valid price!")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\n💡 Analysis:")
    print("   • If prices are close to expected values: ✅ Model trained successfully!")
    print("   • If getting random characters/numbers: ❌ Model needs retraining")
    print("   • If getting very wrong numbers: ⚠️  Try training with more epochs")
    print("\n🔧 If model not working:")
    print("   1. Delete model folder: rm -rf product-price-predictor/")
    print("   2. Regenerate data: python create_data.py")
    print("   3. Train again with improved script")
    print("   4. Run this test again")

if __name__ == "__main__":
    test_model()