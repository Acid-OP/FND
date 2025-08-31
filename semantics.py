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

def split_into_chunks(text, window_size=120, step_size=60):
    """Optimized chunking for fake news detection"""
    # Simple text cleaning
    text = ' '.join(text.split())  # Remove extra whitespace
    words = text.split()
    
    # Return whole text if too short
    if len(words) <= window_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), step_size):
        if i + window_size <= len(words):
            chunk = ' '.join(words[i:i + window_size])
            chunks.append(chunk)
        else:
            # Handle final chunk - only if meaningful size
            final_chunk = ' '.join(words[i:])
            if len(final_chunk.split()) >= window_size // 3:  # At least 1/3 size
                chunks.append(final_chunk)
            break
    
    return chunks

def create_chunk_embeddings(text, article_id, label):
    """Create embeddings for each chunk of an article"""
    chunks = split_into_chunks(text)
    # Batch encode for speed - MAIN OPTIMIZATION
    embeddings = model.encode(chunks, show_progress_bar=False)
    chunk_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_info = {
            'article_id': article_id,
            'chunk_id': i,
            'label': label,
            'chunk_text': chunk,
            'embedding': embedding.tolist()
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

# Save ONLY the JSON file
with open('chunk_embeddings.json', 'w') as f:
    json.dump(all_chunk_embeddings, f, indent=2)

print("✅ Chunk embeddings saved as 'chunk_embeddings.json'")
