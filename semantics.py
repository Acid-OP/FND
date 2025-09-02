import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report

print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

# Load datasets
df_fake = pd.read_csv(r"G:\Lock in\New folder\Dataset\Fake.csv")
df_real = pd.read_csv(r"G:\Lock in\New folder\Dataset\True.csv")

print(f"📊 Dataset sizes:")
print(f"   Fake articles: {len(df_fake):,}")
print(f"   Real articles: {len(df_real):,}")

# Define train/test split
TRAIN_SIZE = 500
TEST_SIZE = 500

def calculate_cosine_similarity(title, body_text):
    """Calculate cosine similarity between title and body embeddings using sliding window chunking"""
    try:
        if not title or not body_text:
            return 0.5
            
        title_embedding = model.encode([str(title)], show_progress_bar=False)[0]
        
        # Sliding window chunking for body text
        body_text_str = str(body_text)
        chunk_size = 300
        overlap = 50
        max_chunks = 5  # Limit chunks for performance
        
        chunks = []
        start = 0
        chunk_count = 0
        
        while start < len(body_text_str) and chunk_count < max_chunks:
            end = min(start + chunk_size, len(body_text_str))
            chunk = body_text_str[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
                chunk_count += 1
            
            start += (chunk_size - overlap)
        
        if not chunks:
            return 0.5
        
        # Get embeddings for all chunks
        chunk_embeddings = model.encode(chunks, show_progress_bar=False)
        
        # Calculate similarities between title and each chunk
        max_similarity = 0.0
        for chunk_embedding in chunk_embeddings:
            similarity = cosine_similarity([title_embedding], [chunk_embedding])[0][0]
            max_similarity = max(max_similarity, similarity)
        
        return float(max_similarity)
    except:
        return 0.5

# STEP 1: Get similarities for training data (first 500 of each)
print(f"\n🚀 Processing training data...")
train_fake = df_fake.iloc[:TRAIN_SIZE]                 # rows 0..499
test_fake  = df_fake.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]   # 500..999

train_real = df_real.iloc[:TRAIN_SIZE]
test_real  = df_real.iloc[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

# Calculate similarities
fake_similarities = []
real_similarities = []

print("Processing fake articles...")
for idx, row in train_fake.iterrows():
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{TRAIN_SIZE}")
    similarity = calculate_cosine_similarity(row['title'], row['text'])
    fake_similarities.append(similarity)

print("Processing real articles...")
for idx, row in train_real.iterrows():
    if idx % 100 == 0:
        print(f"  Progress: {idx}/{TRAIN_SIZE}")
    similarity = calculate_cosine_similarity(row['title'], row['text'])
    real_similarities.append(similarity)

# Convert to arrays
fake_similarities = np.array(fake_similarities)
real_similarities = np.array(real_similarities)

print(f"\n📊 Training similarities:")
print(f"   Fake news - Mean: {fake_similarities.mean():.3f}")
print(f"   Real news - Mean: {real_similarities.mean():.3f}")

# STEP 2: Prepare data for ROC curve
# Combine similarities and create labels
all_similarities = np.concatenate([fake_similarities, real_similarities])
# Labels: 0 = fake, 1 = real (for ROC curve interpretation)
all_labels = np.concatenate([np.zeros(len(fake_similarities)), np.ones(len(real_similarities))])

print(f"\n🔍 Finding optimal threshold using ROC curve...")

# Calculate ROC curve
# Higher similarity should predict real news (label=1)
fpr, tpr, thresholds = roc_curve(all_labels, all_similarities)
roc_auc = auc(fpr, tpr)

# Find optimal threshold using Youden's J statistic (TPR - FPR)
youdens_j = tpr - fpr
optimal_idx = np.argmax(youdens_j)
optimal_threshold = thresholds[optimal_idx]
optimal_tpr = tpr[optimal_idx]
optimal_fpr = fpr[optimal_idx]
optimal_j = youdens_j[optimal_idx]

print(f"✅ OPTIMAL THRESHOLD FROM ROC:")
print(f"   Threshold: {optimal_threshold:.3f}")
print(f"   True Positive Rate: {optimal_tpr:.3f}")
print(f"   False Positive Rate: {optimal_fpr:.3f}")
print(f"   Youden's J: {optimal_j:.3f}")
print(f"   Training ROC AUC: {roc_auc:.3f}")

# Prediction rule: similarity >= threshold = real, similarity < threshold = fake
train_predictions = (all_similarities >= optimal_threshold).astype(int)
train_accuracy = accuracy_score(all_labels, train_predictions)
print(f"   Training accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")

# STEP 3: Test on test dataset
print(f"\n🧪 Processing test data...")
test_fake = df_fake.iloc[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]  
test_real = df_real.iloc[TRAIN_SIZE:TRAIN_SIZE+TEST_SIZE]  

test_fake_similarities = []
test_real_similarities = []

print("Processing fake test articles...")
for idx, row in test_fake.iterrows():
    if (idx - TRAIN_SIZE) % 100 == 0:
        print(f"  Progress: {idx - TRAIN_SIZE}/{TEST_SIZE}")
    similarity = calculate_cosine_similarity(row['title'], row['text'])
    test_fake_similarities.append(similarity)

print("Processing real test articles...")
for idx, row in test_real.iterrows():
    if (idx - TRAIN_SIZE) % 100 == 0:
        print(f"  Progress: {idx - TRAIN_SIZE}/{TEST_SIZE}")
    similarity = calculate_cosine_similarity(row['title'], row['text'])
    test_real_similarities.append(similarity)

# Convert to arrays
test_fake_similarities = np.array(test_fake_similarities)
test_real_similarities = np.array(test_real_similarities)

# Combine test data
test_similarities = np.concatenate([test_fake_similarities, test_real_similarities])
test_labels = np.concatenate([np.zeros(len(test_fake_similarities)), np.ones(len(test_real_similarities))])

# Apply threshold to test data
test_predictions = (test_similarities >= optimal_threshold).astype(int)
test_accuracy = accuracy_score(test_labels, test_predictions)

# Calculate test ROC AUC
test_fpr, test_tpr, _ = roc_curve(test_labels, test_similarities)
test_roc_auc = auc(test_fpr, test_tpr)

# RESULTS
print(f"\n" + "="*60)
print("🎯 FINAL RESULTS")
print("="*60)
print(f"📊 Similarities Analysis:")
print(f"   Training - Fake: {fake_similarities.mean():.3f}, Real: {real_similarities.mean():.3f}")
print(f"   Testing  - Fake: {test_fake_similarities.mean():.3f}, Real: {test_real_similarities.mean():.3f}")

print(f"\n📊 Performance:")
print(f"   Optimal threshold: {optimal_threshold:.3f}")
print(f"   Training accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
print(f"   Test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
print(f"   Test ROC AUC: {test_roc_auc:.3f}")

# # VISUALIZATION: ROC Curve with coordinates
# plt.figure(figsize=(15, 5))

# # 2. ROC Curve with optimal point
# plt.subplot(1, 3, 2)
# plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Training ROC (AUC={roc_auc:.3f})')
# plt.plot(test_fpr, test_tpr, 'r-', linewidth=2, label=f'Test ROC (AUC={test_roc_auc:.3f})')
# plt.plot(optimal_fpr, optimal_tpr, 'go', markersize=10, 
#          label=f'Optimal Point\n({optimal_fpr:.3f}, {optimal_tpr:.3f})')
# plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve Analysis')
# plt.legend()
# plt.grid(True, alpha=0.3)

# # 3. Youden's J optimization
# plt.subplot(1, 3, 3)
# plt.plot(thresholds, youdens_j, 'g-', linewidth=2, label="Youden's J = TPR - FPR")
# plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
#             label=f'Optimal Threshold: {optimal_threshold:.3f}')
# plt.plot(optimal_threshold, optimal_j, 'ro', markersize=8, 
#          label=f'Max J = {optimal_j:.3f}')
# plt.xlabel('Threshold Value')
# plt.ylabel("Youden's J Statistic")
# plt.title('Threshold Optimization')
# plt.legend()
# plt.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('roc_analysis_clean.png', dpi=300, bbox_inches='tight')
# plt.show()

print(f"\n✅ Analysis complete!")
print(f"📊 Key insight: Real news has higher title-body similarity ({real_similarities.mean():.3f}) than fake news ({fake_similarities.mean():.3f})")
print(f"🎯 Threshold {optimal_threshold:.3f} gives {test_accuracy*100:.1f}% accuracy on test data")