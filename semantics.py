import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Load datasets
print("Loading ISOT datasets...")
df_fake = pd.read_csv(r"G:\Lock in\New folder\Dataset\Fake.csv")
df_real = pd.read_csv(r"G:\Lock in\New folder\Dataset\True.csv")

# Take small sample
fake_sample = df_fake.head(5)
real_sample = df_real.head(5)

def sliding_window_chunking(text, window_size=100, step_size=50):
    """BEST FOR FND: Dense overlapping coverage"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words) - window_size + 1, step_size):
        chunk = ' '.join(words[i:i + window_size])
        chunks.append(chunk)
    
    return chunks

def create_chunk_embeddings(text, article_id, label):
    """Create embeddings for each chunk of an article"""
    chunks = sliding_window_chunking(text, window_size=100, step_size=50)
    
    chunk_data = []
    for i, chunk in enumerate(chunks):
        # Create embedding for this chunk
        embedding = model.encode([chunk])[0]  # Get single embedding vector
        
        # Convert to simple list of numbers
        embedding_list = embedding.tolist()
        
        chunk_info = {
            'article_id': article_id,
            'chunk_id': i,
            'label': label,  # 1=fake, 0=real
            'chunk_text': chunk,
            'embedding': embedding_list  # This is your [1,2,3,...] format
        }
        chunk_data.append(chunk_info)
    
    return chunk_data

# Process all articles and create chunk embeddings
print("Creating chunk embeddings...")
all_chunk_embeddings = []

# Process fake articles
for idx, row in fake_sample.iterrows():
    print(f"Processing fake article {idx+1}/5...")
    chunks = create_chunk_embeddings(row['text'], f"fake_{idx}", label=1)
    all_chunk_embeddings.extend(chunks)

# Process real articles  
for idx, row in real_sample.iterrows():
    print(f"Processing real article {idx+1}/5...")
    chunks = create_chunk_embeddings(row['text'], f"real_{idx}", label=0)
    all_chunk_embeddings.extend(chunks)

print(f"Total chunks created: {len(all_chunk_embeddings)}")

# Save ONLY the JSON file (complete data)
with open('chunk_embeddings.json', 'w') as f:
    json.dump(all_chunk_embeddings, f, indent=2)

print("✅ Chunk embeddings saved as 'chunk_embeddings.json'")

# Show what we created
print("\n" + "="*50)
print("EMBEDDING DETAILS")
print("="*50)
print(f"Total chunks: {len(all_chunk_embeddings)}")
print(f"Embedding dimension: {len(all_chunk_embeddings[0]['embedding'])}")

# Show sample embedding (first 10 numbers)
sample_embedding = all_chunk_embeddings[0]['embedding'][:10]
print(f"\nSample embedding (first 10 values): {sample_embedding}")
print(f"Full embedding has {len(all_chunk_embeddings[0]['embedding'])} numbers")

# Show file size
import os
json_size = os.path.getsize('chunk_embeddings.json') / 1024 / 1024
print(f"\nFile size: chunk_embeddings.json: {json_size:.2f} MB")

print("\n" + "="*50)
print("ONE FILE CREATED:")
print("="*50)
print("📄 chunk_embeddings.json - Complete data with text + embeddings")

print("\n✅ SIMPLIFIED CHUNK EMBEDDINGS STORAGE COMPLETE!")

# BONUS: Show how to extract arrays from JSON when you need them
print("\n" + "="*50)
print("HOW TO USE THE JSON FILE:")
print("="*50)
print("# When you need just the vectors:")
print("with open('chunk_embeddings.json', 'r') as f:")
print("    data = json.load(f)")
print("embeddings = np.array([chunk['embedding'] for chunk in data])")
print("labels = np.array([chunk['label'] for chunk in data])")
print("# Now you have your arrays!")