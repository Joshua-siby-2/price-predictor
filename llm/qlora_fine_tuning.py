import os
import torch
import pandas as pd
import logging
import argparse
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fine_tuning.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Use GPT2 for better results
model_name = "gpt2"

def create_prompt(sample):
    """Create a properly formatted prompt with EOS token - NO DOLLAR SIGNS"""
    price = int(float(str(sample['price']).replace('$', '').replace(',', '').strip()))
    
    # Simple format with EOS token
    return f"""Product: {sample['product_name']}
Description: {sample['product_description']}
Price: {price}<|endoftext|>"""

def tokenize_function(examples, tokenizer):
    """Tokenize with proper truncation"""
    result = tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=128,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

def load_and_prepare_dataset(tokenizer, dataset_path):
    """Load and prepare the dataset from CSV"""
    logging.info("Loading and preparing dataset...")
    try:
        df = pd.read_csv(dataset_path)
        logging.info(f"Loaded dataset with {len(df)} samples")
        
        if len(df) < 50:
            logging.warning(f"Dataset is very small ({len(df)} samples). Minimum recommended: 100+ samples")
        
        required_columns = ['product_name', 'product_description', 'price']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        logging.info("Cleaning and formatting the price column...")
        def clean_price(price):
            if pd.isna(price):
                return 0
            price_str = str(price).replace('$', '').replace(',', '').strip()
            try:
                return int(float(price_str))
            except ValueError:
                return 0
        
        df['price'] = df['price'].apply(clean_price)
        df = df.dropna(subset=['product_name', 'product_description'])
        df = df[df['product_name'].str.strip() != '']
        df = df[df['product_description'].str.strip() != '']
        df = df[df['price'] > 0]
        
        logging.info(f"Dataset after cleaning: {len(df)} samples")
        logging.info(f"Price range: {df['price'].min()} - {df['price'].max()}")
        
        # Create prompts
        logging.info("Creating prompts with EOS tokens (no $ signs)...")
        texts = [create_prompt(row) for _, row in df.iterrows()]
        
        dataset_dict = {"text": texts}
        dataset = Dataset.from_dict(dataset_dict)
        
        logging.info("Sample training examples:")
        for i in range(min(3, len(dataset))):
            logging.info(f"\nExample {i+1}:")
            logging.info(dataset[i]['text'])
            logging.info("-" * 50)
        
        # Tokenize dataset
        logging.info("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=1
        )
        
        return tokenized_dataset
        
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for product price prediction.")
    parser.add_argument("--dataset_path", type=str, 
                    default="data/product_data.csv")
    parser.add_argument("--output_dir", type=str, 
                    default="logs/results")
    parser.add_argument("--new_model", type=str, 
                    default="models/product-price-predictor")
    args = parser.parse_args()

    logging.info("Starting fine-tuning process...")
    
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")
    
    # Load tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(tokenizer, args.dataset_path)
    
    # OPTIMIZED LoRA configuration
    lora_r = 8  # Slightly higher for better learning
    lora_alpha = 16
    lora_dropout = 0.05
    logging.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Training parameters - BALANCED for quality and speed
    num_train_epochs = 5  # More epochs for better learning
    fp16 = False
    bf16 = False
    
    if cuda_available:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            bf16 = True
            logging.info("Using bf16")
        else:
            fp16 = True
            logging.info("Using fp16")
    
    # Training configuration
    per_device_train_batch_size = 4 if cuda_available else 2
    gradient_accumulation_steps = 2  # Effective batch size = 8 (GPU) or 4 (CPU)
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4  # Standard learning rate
    weight_decay = 0.01
    optim = "adamw_torch"
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.1  # More warmup for stable training
    save_steps = 500
    logging_steps = 10
    
    # Quantization configuration
    bnb_config = None
    if cuda_available:
        logging.info("Setting up 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    
    logging.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if cuda_available else None,
        trust_remote_code=True,
        torch_dtype=torch.float32 if not cuda_available else torch.float16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    if cuda_available and bnb_config:
        model = prepare_model_for_kbit_training(model)
    
    logging.info("Loading LoRA configuration...")
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "wte", "wpe"],  # GPT2 attention modules
    )
    
    logging.info("Applying LoRA to model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    logging.info("Setting training arguments...")
    training_arguments = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",
        save_total_limit=1,
        gradient_checkpointing=gradient_checkpointing,
        save_strategy="epoch",
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        logging_first_step=True,
    )
    
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    logging.info("Starting training...")
    logging.info("=" * 80)
    trainer.train()
    
    logging.info("Saving model...")
    trainer.model.save_pretrained(args.new_model)
    tokenizer.save_pretrained(args.new_model)
    
    logging.info(f"Model saved to {args.new_model}")
    logging.info("=" * 80)
    logging.info("Training completed successfully!")
    logging.info("You can now use the /predict endpoint to make predictions")

if __name__ == "__main__":
    main()
