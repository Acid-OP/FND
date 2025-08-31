import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_curve, auc

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Load datasets
print("Loading ISOT datasets...")
df_fake = pd.read_csv(r"G:\Lock in\New folder\Dataset\Fake.csv")
df_real = pd.read_csv(r"G:\Lock in\New folder\Dataset\True.csv")

# Increase sample size for better analysis
SAMPLE_SIZE = 50  # Increased for better patterns
fake_sample = df_fake.head(SAMPLE_SIZE)
real_sample = df_real.head(SAMPLE_SIZE)

print(f"Using {SAMPLE_SIZE} articles from each dataset for analysis")

def split_into_chunks(text, window_size=120, step_size=60):
    """Optimized chunking for fake news detection"""
    text = ' '.join(text.split())
    words = text.split()
    
    if len(words) <= window_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), step_size):
        if i + window_size <= len(words):
            chunk = ' '.join(words[i:i + window_size])
            chunks.append(chunk)
        else:
            final_chunk = ' '.join(words[i:])
            if len(final_chunk.split()) >= window_size // 3:
                chunks.append(final_chunk)
            break
    
    return chunks

def calculate_title_body_similarity(title, body_text):
    """Calculate cosine similarity between title and body"""
    # Get embeddings
    title_emb = model.encode([title], show_progress_bar=False)[0]
    
    # For body, take average of all chunk embeddings
    chunks = split_into_chunks(body_text)
    body_embeddings = model.encode(chunks, show_progress_bar=False)
    body_avg_emb = np.mean(body_embeddings, axis=0)  # Average all chunks
    
    # Calculate cosine similarity
    similarity = cosine_similarity([title_emb], [body_avg_emb])[0][0]
    return similarity

# Calculate similarities for all articles
print("Calculating title-body similarities...")
similarities_data = []

# Process fake articles
print("Processing fake articles...")
for idx, row in fake_sample.iterrows():
    print(f"  Fake article {idx+1}/{SAMPLE_SIZE}")
    similarity = calculate_title_body_similarity(row['title'], row['text'])
    similarities_data.append({
        'article_id': f"fake_{idx}",
        'label': 1,  # fake
        'title': row['title'],
        'similarity': similarity
    })

# Process real articles
print("Processing real articles...")
for idx, row in real_sample.iterrows():
    print(f"  Real article {idx+1}/{SAMPLE_SIZE}")
    similarity = calculate_title_body_similarity(row['title'], row['text'])
    similarities_data.append({
        'article_id': f"real_{idx}",
        'label': 0,  # real
        'title': row['title'],
        'similarity': similarity
    })

# Convert to arrays for analysis
similarities = np.array([d['similarity'] for d in similarities_data])
labels = np.array([d['label'] for d in similarities_data])

fake_similarities = similarities[labels == 1]
real_similarities = similarities[labels == 0]

print(f"\n" + "="*60)
print("SIMILARITY ANALYSIS RESULTS")
print("="*60)

print(f"📊 FAKE NEWS SIMILARITIES:")
print(f"   Average: {fake_similarities.mean():.3f}")
print(f"   Range: {fake_similarities.min():.3f} to {fake_similarities.max():.3f}")

print(f"\n📊 REAL NEWS SIMILARITIES:")
print(f"   Average: {real_similarities.mean():.3f}")
print(f"   Range: {real_similarities.min():.3f} to {real_similarities.max():.3f}")

print(f"\n🎯 DIFFERENCE:")
print(f"   Real - Fake = {real_similarities.mean() - fake_similarities.mean():.3f}")

# Plot the distributions
plt.figure(figsize=(12, 8))

# Plot 1: Overlapping histograms
plt.subplot(2, 2, 1)
plt.hist(fake_similarities, bins=20, alpha=0.7, label='Fake News', color='red', density=True)
plt.hist(real_similarities, bins=20, alpha=0.7, label='Real News', color='green', density=True)
plt.xlabel('Title-Body Cosine Similarity')
plt.ylabel('Density')
plt.title('Similarity Distributions: Fake vs Real')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Box plots
plt.subplot(2, 2, 2)
plt.boxplot([fake_similarities, real_similarities], labels=['Fake', 'Real'])
plt.ylabel('Cosine Similarity')
plt.title('Similarity Box Plots')
plt.grid(True, alpha=0.3)

# Find optimal threshold using multiple values
thresholds = np.linspace(similarities.min(), similarities.max(), 100)
accuracies = []

for threshold in thresholds:
    predictions = (similarities >= threshold).astype(int)  # 1 if >= threshold (real), 0 if < threshold (fake)
    # Note: We predict REAL if similarity >= threshold, FAKE if < threshold
    accuracy = accuracy_score(labels, predictions)
    accuracies.append(accuracy)

best_threshold_idx = np.argmax(accuracies)
best_threshold = thresholds[best_threshold_idx]
best_accuracy = accuracies[best_threshold_idx]

# Plot 3: Threshold analysis
plt.subplot(2, 2, 3)
plt.plot(thresholds, accuracies, 'b-', linewidth=2)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best: {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Threshold vs Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: ROC Curve
plt.subplot(2, 2, 4)
# For ROC, we need probabilities - use similarities as scores
fpr, tpr, roc_thresholds = roc_curve(labels, similarities)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fnd_similarity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Final results
print(f"\n" + "="*60)
print("🎯 FINAL FND RESEARCH RESULTS")
print("="*60)

print(f"📈 BEST THRESHOLD: {best_threshold:.3f}")
print(f"📈 BEST ACCURACY: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
print(f"📈 ROC AUC SCORE: {roc_auc:.3f}")

# Test the threshold
final_predictions = (similarities >= best_threshold).astype(int)
fake_correct = np.sum((labels == 1) & (final_predictions == 1))  # Correctly identified real
real_correct = np.sum((labels == 0) & (final_predictions == 0))  # Correctly identified fake

print(f"\n📊 DETECTION BREAKDOWN:")
print(f"   Fake articles correctly identified: {SAMPLE_SIZE - fake_correct}/{SAMPLE_SIZE}")
print(f"   Real articles correctly identified: {real_correct}/{SAMPLE_SIZE}")

# Show some examples
print(f"\n🔍 EXAMPLE PREDICTIONS:")
for i in range(5):
    data = similarities_data[i]
    pred = "REAL" if similarities[i] >= best_threshold else "FAKE"
    actual = "FAKE" if data['label'] == 1 else "REAL"
    sim = similarities[i]
    
    status = "✅" if pred == actual else "❌"
    print(f"   {status} Similarity: {sim:.3f} → Predicted: {pred}, Actual: {actual}")

print(f"\n🚀 PRODUCTION DETECTOR:")
print(f"def detect_fake_news(title, body):")
print(f"    # Calculate title-body similarity")
print(f"    # If similarity < {best_threshold:.3f}: return 'FAKE'")
print(f"    # If similarity >= {best_threshold:.3f}: return 'REAL'")

# Save analysis results
analysis_results = {
    'best_threshold': float(best_threshold),
    'best_accuracy': float(best_accuracy),
    'roc_auc': float(roc_auc),
    'fake_avg_similarity': float(fake_similarities.mean()),
    'real_avg_similarity': float(real_similarities.mean()),
    'similarities_data': similarities_data
}

with open('fnd_analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print(f"\n✅ Analysis results saved as 'fnd_analysis_results.json'")
print(f"✅ Plots saved as 'fnd_similarity_analysis.png'")