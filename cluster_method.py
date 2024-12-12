from transformers import pipeline
import pandas as pd
import random
import json
import os
from tqdm import tqdm
import warnings
import logging

# Setup
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct", device=0, max_new_tokens=256)
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

# Load data and create all prompts first
curated = [item for sublist in pd.read_csv('../data/LLM_clean_curated.csv').values.tolist() for item in sublist]

random_samples = [
    random.sample(curated, 20) 
    for _ in range(1_000_000)
]

# Format all prompts
formatted_prompts = []
for sample in random_samples:
    sample_text = ", ".join(sample)
    prompt = [
        {"role": "system", "content": "respond with nothing else other than a numbered list of comma seperated words, no other words, no explanations, nothing else. Return the words exactly as they come, no captialisaion or anything else."},
        {"role": "user", "content": f"List of words: {sample_text}. Cluster these random words by semanticity:"}
    ]
    formatted_prompts.append(prompt)

# Process in batches and save periodically
BATCH_SIZE = 32
SAVE_INTERVAL = 10_000
outputs = []

total_batches = len(formatted_prompts) // BATCH_SIZE + (1 if len(formatted_prompts) % BATCH_SIZE != 0 else 0)

for i in tqdm(range(0, len(formatted_prompts), BATCH_SIZE), total=total_batches, desc="Processing batches"):
    batch = formatted_prompts[i:i+BATCH_SIZE]
    batch_outputs = pipe(
        batch,
        batch_size=BATCH_SIZE,
        max_new_tokens=256
    )
    outputs.extend(batch_outputs)
    
    # Save every SAVE_INTERVAL samples
    if len(outputs) >= SAVE_INTERVAL:
        if os.path.exists('cluster_words3.json'):
            with open('cluster_words3.json', 'r') as f:
                existing_outputs = json.load(f)
            outputs = existing_outputs + outputs
            
        with open('cluster_words3.json', 'w') as f:
            json.dump(outputs, f, indent=2)
        
        print(f"Saved {len(outputs)} outputs")
        outputs = []  # Clear memory after saving

# Save any remaining outputs
if outputs:
    if os.path.exists('cluster_words3.json'):
        with open('cluster_words3.json', 'r') as f:
            existing_outputs = json.load(f)
        outputs = existing_outputs + outputs
        
    with open('cluster_words3.json', 'w') as f:
        json.dump(outputs, f, indent=2)

print("Processing complete")