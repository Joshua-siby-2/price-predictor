import os
import torch
import pandas as pd
import logging
import argparse
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fine_tuning.log"),
        logging.StreamHandler()
    ]
)

# Model and dataset parameters
model_name = "distilgpt2"

def create_prompt(sample):
    """Create a properly formatted prompt for training - SIMPLIFIED"""
    price = int(float(str(sample['price']).replace('$', '').replace(',', '').strip()))
    
    # Simplified format - clear instruction-response format
    return f"""Product: {sample['product_name']}
Description: {sample['product_description']}
Price: ${price}"""

def tokenize_function(examples, tokenizer):
    """Tokenize the text with reduced max_length for faster training"""
    return tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")

def load_and_prepare_dataset(tokenizer, dataset_path):
    """Load and prepare the dataset from CSV"""
    logging.info("Loading and preparing dataset...")
    try:
        # Load CSV file
        df = pd.read_csv(dataset_path)
        logging.info(f"Loaded dataset with {len(df)} samples")
        
        # Check dataset size
        if len(df) < 50:
            logging.warning(f"Dataset is very small ({len(df)} samples). Consider adding more data for better results.")
            logging.warning("Minimum recommended: 100-500 samples")
        
        # Check if required columns exist
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
        
        # Remove invalid rows
        logging.info("Removing rows with invalid data...")
        df = df.dropna(subset=['product_name', 'product_description'])
        df = df[df['product_name'].str.strip() != '']
        df = df[df['product_description'].str.strip() != '']
        df = df[df['price'] > 0]
        
        logging.info(f"Dataset after cleaning: {len(df)} samples")
        logging.info(f"Price range: ${df['price'].min()} - ${df['price'].max()}")
        
        # Create prompts
        logging.info("Creating prompts...")
        texts = []
        for _, row in df.iterrows():
            texts.append(create_prompt(row))
        
        # Create dataset
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
            num_proc=os.cpu_count()  # Parallel processing
        )
        
        return tokenized_dataset
        
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for product price prediction.")
    parser.add_argument("--dataset_path", type=str, default="data/product_data.csv", help="Path to the training dataset.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the training results.")
    parser.add_argument("--new_model", type=str, default="product-price-predictor", help="Path to save the fine-tuned model.")
    args = parser.parse_args()

    logging.info("Starting fine-tuning process...")
    
    # Check for CUDA availability
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA available: {cuda_available}")
    
    # Load tokenizer
    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(tokenizer, args.dataset_path)
    
    # OPTIMIZED LoRA configuration - smaller rank for faster training
    lora_r = 8  # Reduced from 16
    lora_alpha = 16  # Reduced from 32
    lora_dropout = 0.05
    logging.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # bitsandbytes parameters
    use_4bit = cuda_available
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    
    # OPTIMIZED Training parameters for SPEED
    num_train_epochs = 10  # Reduced from 5/10
    fp16 = False
    bf16 = False
    
    if cuda_available:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            logging.info("GPU supports bfloat16, enabling bf16.")
            bf16 = True
        else:
            logging.info("GPU does not support bfloat16, enabling fp16.")
            fp16 = True
    
    # OPTIMIZED batch sizes for faster training
    per_device_train_batch_size = 2 if not cuda_available else 4  # Increased
    gradient_accumulation_steps = 1  # Reduced to 1 for both
    gradient_checkpointing = False  # Disabled for speed
    max_grad_norm = 0.3
    learning_rate = 2e-4  # Slightly higher for faster convergence
    weight_decay = 0.001
    optim = "adamw_torch"  # Faster than paged_adamw
    lr_scheduler_type = "constant"  # Simpler scheduler
    max_steps = -1
    warmup_ratio = 0.03  # Reduced warmup
    save_steps = 1000  # Less frequent saves
    logging_steps = 10
    
    # Quantization configuration
    if use_4bit:
        logging.info("Setting up 4-bit quantization...")
        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )
    else:
        bnb_config = None
    
    logging.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if cuda_available else None,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Explicit dtype
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Prepare model for k-bit training if using quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    logging.info("Loading LoRA configuration...")
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "q_proj", "v_proj"],  # DistilGPT2 modules
    )
    
    # Apply LoRA to model
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
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=False,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",  # Disable reporting for speed
        save_total_limit=1,  # Keep only 1 checkpoint
        dataloader_num_workers=os.cpu_count(),  # Parallel data loading
        gradient_checkpointing=gradient_checkpointing,
        save_strategy="epoch",  # Save checkpoints every epoch
    )
    
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    logging.info("Starting training...")
    trainer.train()
    
    logging.info("Saving model...")
    trainer.model.save_pretrained(args.new_model)
    tokenizer.save_pretrained(args.new_model)
    
    logging.info(f"Model saved to {args.new_model}")
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()