import pandas as pd
import random

def generate_focused_laptop_data(num_samples=150):
    """Generate focused laptop data with VERY clear, linear patterns - NO $ SIGNS"""
    products = []
    
    # Simplified specs for crystal-clear learning
    ram_options = [6, 16]
    storage_options = [128, 512]
    # processors = {
    #     'Intel i5': 600,
    #     'AMD Ryzen 5': 200
    # }
    
    for i in range(num_samples):
        ram = random.choice(ram_options)
        storage = random.choice(storage_options)
        # processor = random.choice(list(processors.keys()))
        
        # CRYSTAL CLEAR pricing formula - NO randomness for initial learning
        base_price = 6000
        base_price += (ram - 8) * 500  # Each 8GB RAM = +$50
        base_price += (storage - 256) * 1  # Each GB storage = +$0.3
        # base_price += processors[processor]
        
        # Round to nearest 10 for cleaner numbers
        price = int(round(base_price / 10) * 10)
        
        description = f"Laptop with {ram}GB RAM, {storage}GB SSD"
        
        products.append({
            'product_name': 'Laptop',
            'product_description': description,
            'price': price  # Just the number, no $ sign
        })
    
    return pd.DataFrame(products)

def generate_focused_smartphone_data(num_samples=150):
    """Generate focused smartphone data with VERY clear, linear patterns - NO $ SIGNS"""
    products = []
    
    # Simplified specs
    ram_options = [6, 16]
    storage_options = [128, 512]
    
    for i in range(num_samples):
        ram = random.choice(ram_options)
        storage = random.choice(storage_options)
        
        # CRYSTAL CLEAR pricing
        base_price = 300
        base_price += (ram - 6) * 500  # Each 10GB RAM upgrade = +$40
        base_price += (storage - 128) * 1  # Each GB storage = +$0.5
        
        # Round to nearest 10
        price = int(round(base_price / 10) * 10)
        
        description = f"Smartphone with {ram}GB RAM, {storage}GB storage"
        
        products.append({
            'product_name': 'Smartphone',
            'product_description': description,
            'price': price  # Just the number, no $ sign
        })
    
    return pd.DataFrame(products)

def main():
    """Generate focused training dataset - 150 samples for better learning - NO $ SIGNS"""
    print("Generating focused training data for efficient LLM learning...")
    print("Creating 150 samples with CRYSTAL CLEAR patterns (NO $ SIGNS)...\n")
    
    # Generate focused data
    laptop_df = generate_focused_laptop_data(150)
    smartphone_df = generate_focused_smartphone_data(150)
    
    # Combine all data
    combined_df = pd.concat([laptop_df, smartphone_df], ignore_index=True)
    
    # Shuffle the data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Generated {len(combined_df)} focused training samples")
    print(f"   Price range: {combined_df['price'].min()} - {combined_df['price'].max()}")
    print(f"\n📊 Product distribution:")
    print(combined_df['product_name'].value_counts())
    
    # Show price ranges by product type
    print(f"\n📈 Price ranges:")
    laptop_prices = combined_df[combined_df['product_name'] == 'Laptop']['price']
    phone_prices = combined_df[combined_df['product_name'] == 'Smartphone']['price']
    
    print(f"   Laptops: {laptop_prices.min()} - {laptop_prices.max()}")
    print(f"   Smartphones: {phone_prices.min()} - {phone_prices.max()}")
    
    # Show the EXACT patterns
    print(f"\n🎯 EXACT PRICING PATTERNS (numbers only, no $ signs):")
    print("\n   LAPTOPS:")
    print(f"   • 8GB RAM, 256GB SSD, Intel i5: ~{2000 + 600}")
    print(f"   • 8GB RAM, 256GB SSD, AMD Ryzen 5: ~{2000 + 200}")
    print(f"   • 16GB RAM, 512GB SSD, Intel i5: ~{2000 + 50 + 77 + 600}")
    print(f"   • 16GB RAM, 512GB SSD, AMD Ryzen 5: ~{2000 + 50 + 77 + 200}")
    
    print("\n   SMARTPHONES:")
    print(f"   • 6GB RAM, 128GB storage: ~{300}")
    print(f"   • 6GB RAM, 512GB storage: ~{300 + 192}")
    print(f"   • 16GB RAM, 128GB storage: ~{300 + 400}")
    print(f"   • 16GB RAM, 512GB storage: ~{300 + 400 + 192}")
    
    # Save to CSV
    output_path = ".\data\product_data.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n💾 Data saved to {output_path}")
    
    # Show actual examples
    print("\n🔍 Sample training data (numbers only):")
    sample_indices = [0, 1, 80, 81]  # Show laptop and phone examples
    for idx in sample_indices:
        if idx < len(combined_df):
            row = combined_df.iloc[idx]
            print(f"\n   {row['product_name']}: {row['product_description']}")
            print(f"   Price: {row['price']}")  # No $ sign
    
    print("\n✅ Benefits of removing $ sign:")
    print("   • Simpler for model to learn (just numbers)")
    print("   • Fewer tokens (faster training)")
    print("   • Less ambiguity in parsing")
    print("   • More consistent output format")
    
    print("\n✅ 150 samples with zero randomness = perfect learning!")
    print("💡 Clear patterns + minimal noise + no $ = faster learning")
    print("⚡ Expected training time: 5-10 minutes (CPU) / 2-3 minutes (GPU)")

if __name__ == "__main__":
    main()