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

def create_title_embeddings(title, article_id, label):
    """Create embeddings for title (no chunking needed for titles)"""
    # Clean title
    title = ' '.join(str(title).split())
    
    # Create single embedding for title
    embedding = model.encode([title], show_progress_bar=False)[0]
    
    title_data = {
        'article_id': article_id,
        'content_type': 'title',
        'label': label,
        'text': title,
        'embedding': embedding.tolist()
    }
    
    return title_data

def create_body_chunk_embeddings(text, article_id, label):
    """Create embeddings for each chunk of article body"""
    chunks = split_into_chunks(text)
    # Batch encode for speed
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    chunk_data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_info = {
            'article_id': article_id,
            'chunk_id': i,
            'content_type': 'body_chunk',
            'label': label,
            'text': chunk,
            'embedding': embedding.tolist()
        }
        chunk_data.append(chunk_info)
    
    return chunk_data

# Storage containers
all_title_embeddings = []
all_body_embeddings = []

print("Creating separate title and body embeddings...")

# Process fake articles
for idx, row in fake_sample.iterrows():
    print(f"Processing fake article {idx+1}/5...")
    
    # Process title
    title_emb = create_title_embeddings(row['title'], f"fake_{idx}", label=1)
    all_title_embeddings.append(title_emb)
    
    # Process body chunks
    body_chunks = create_body_chunk_embeddings(row['text'], f"fake_{idx}", label=1)
    all_body_embeddings.extend(body_chunks)

# Process real articles  
for idx, row in real_sample.iterrows():
    print(f"Processing real article {idx+1}/5...")
    
    # Process title
    title_emb = create_title_embeddings(row['title'], f"real_{idx}", label=0)
    all_title_embeddings.append(title_emb)
    
    # Process body chunks
    body_chunks = create_body_chunk_embeddings(row['text'], f"real_{idx}", label=0)
    all_body_embeddings.extend(body_chunks)

print(f"Total title embeddings: {len(all_title_embeddings)}")
print(f"Total body chunk embeddings: {len(all_body_embeddings)}")

# Save title embeddings separately
with open('title_embeddings.json', 'w') as f:
    json.dump(all_title_embeddings, f, indent=2)

# Save body embeddings separately  
with open('body_embeddings.json', 'w') as f:
    json.dump(all_body_embeddings, f, indent=2)

print("✅ Title embeddings saved as 'title_embeddings.json'")
print("✅ Body embeddings saved as 'body_embeddings.json'")

# Show what we created
print("\n" + "="*50)
print("SEPARATE EMBEDDINGS CREATED:")
print("="*50)
print(f"📰 Titles: {len(all_title_embeddings)} embeddings")
print(f"📄 Body chunks: {len(all_body_embeddings)} embeddings")

if all_title_embeddings:
    print(f"Title embedding dimension: {len(all_title_embeddings[0]['embedding'])}")
if all_body_embeddings:
    print(f"Body embedding dimension: {len(all_body_embeddings[0]['embedding'])}")

print("\n🎯 FILES FOR FND COMPARISON:")
print("📰 title_embeddings.json - All article titles")
print("📄 body_embeddings.json - All chunked article bodies")
print("\n✅ Ready for separate title vs body analysis!")