import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import re
import seaborn as sns
import time

# Simple readability functions
def simple_flesch_reading_ease(text):
    sentences = len(re.split(r'[.!?]+', text))
    words = len(text.split())
    syllables = sum([len(re.findall(r'[aeiouAEIOU]', word)) for word in text.split()])
    
    if sentences == 0 or words == 0:
        return 50.0
    
    avg_sentence_length = words / sentences
    avg_syllables_per_word = syllables / words
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return max(0, min(100, flesch_score))

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Load datasets
print("Loading ISOT datasets...")
df_fake = pd.read_csv(r"G:\Lock in\New folder\Dataset\Fake.csv")
df_real = pd.read_csv(r"G:\Lock in\New folder\Dataset\True.csv")

print(f"📊 FULL DATASET SIZES:")
print(f"   Fake articles: {len(df_fake):,}")
print(f"   Real articles: {len(df_real):,}")

# REASONABLE SAMPLE SIZES - Much more manageable!
TRAIN_SIZE_PER_CLASS = 500  # 500 fake + 500 real = 1000 training
TEST_SIZE_PER_CLASS = 200   # 200 fake + 200 real = 400 testing

print(f"\n🎯 REASONABLE SAMPLE SIZES:")
print(f"   Training: {TRAIN_SIZE_PER_CLASS} fake + {TRAIN_SIZE_PER_CLASS} real = {TRAIN_SIZE_PER_CLASS*2:,} total")
print(f"   Testing:  {TEST_SIZE_PER_CLASS} fake + {TEST_SIZE_PER_CLASS} real = {TEST_SIZE_PER_CLASS*2:,} total")
print(f"   Total to process: {(TRAIN_SIZE_PER_CLASS + TEST_SIZE_PER_CLASS)*2:,} articles")

# Randomly sample articles
np.random.seed(42)
fake_sample = df_fake.sample(n=TRAIN_SIZE_PER_CLASS + TEST_SIZE_PER_CLASS, random_state=42).reset_index(drop=True)
real_sample = df_real.sample(n=TRAIN_SIZE_PER_CLASS + TEST_SIZE_PER_CLASS, random_state=42).reset_index(drop=True)

# Split into train/test
fake_train = fake_sample.iloc[:TRAIN_SIZE_PER_CLASS]
fake_test = fake_sample.iloc[TRAIN_SIZE_PER_CLASS:]
real_train = real_sample.iloc[:TRAIN_SIZE_PER_CLASS]
real_test = real_sample.iloc[TRAIN_SIZE_PER_CLASS:]

def split_into_chunks(text, window_size=120, step_size=60):
    """Optimized chunking"""
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

def calculate_features_fast(title, body_text):
    """Streamlined feature calculation"""
    try:
        # Core semantic features
        title_emb = model.encode([title], show_progress_bar=False)[0]
        
        # Use only first 1000 characters of body for speed
        body_sample = body_text[:1000] if len(body_text) > 1000 else body_text
        body_emb = model.encode([body_sample], show_progress_bar=False)[0]
        basic_similarity = cosine_similarity([title_emb], [body_emb])[0][0]
        
        # Quick linguistic features
        title_word_count = len(title.split())
        emotional_words = ['shocking', 'breaking', 'urgent', 'crisis', 'scandal', 'exclusive']
        emotional_score = sum(1 for word in emotional_words if word.lower() in title.lower())
        caps_ratio = sum(1 for c in title if c.isupper()) / len(title) if title else 0
        exclamation_count = title.count('!')
        
        # Basic readability (simplified)
        body_readability = simple_flesch_reading_ease(body_sample)
        
        return {
            'basic_similarity': float(basic_similarity),
            'title_word_count': title_word_count,
            'emotional_score': emotional_score,
            'caps_ratio': caps_ratio,
            'exclamation_count': exclamation_count,
            'body_readability': float(body_readability)
        }
    except Exception as e:
        print(f"Error processing: {e}")
        return {
            'basic_similarity': 0.5,
            'title_word_count': 10,
            'emotional_score': 0,
            'caps_ratio': 0.1,
            'exclamation_count': 0,
            'body_readability': 50.0
        }

# Process TRAINING data
print(f"\n🚀 PROCESSING TRAINING DATA...")
start_time = time.time()
train_features_data = []

print("Processing fake training articles...")
for idx, row in fake_train.iterrows():
    if idx % 50 == 0:
        print(f"  Fake progress: {idx}/{TRAIN_SIZE_PER_CLASS}")
    features = calculate_features_fast(row['title'], row['text'])
    features.update({'label': 1, 'type': 'fake'})  # fake = 1
    train_features_data.append(features)

print("Processing real training articles...")
for idx, row in real_train.iterrows():
    if idx % 50 == 0:
        print(f"  Real progress: {idx}/{TRAIN_SIZE_PER_CLASS}")
    features = calculate_features_fast(row['title'], row['text'])
    features.update({'label': 0, 'type': 'real'})  # real = 0
    train_features_data.append(features)

training_time = time.time() - start_time
print(f"✅ Training completed in {training_time:.1f} seconds")

# Convert to DataFrame
train_df = pd.DataFrame(train_features_data)
feature_columns = ['basic_similarity', 'title_word_count', 'emotional_score', 
                  'caps_ratio', 'exclamation_count', 'body_readability']

# Train models
print(f"\n🤖 TRAINING MODELS...")
feature_matrix = train_df[feature_columns].values
train_labels = train_df['label'].values

print(f"Training data distribution:")
print(f"   Fake (label=1): {sum(train_labels == 1):,}")
print(f"   Real (label=0): {sum(train_labels == 0):,}")

# Scale features
scaler = StandardScaler()
feature_matrix_scaled = scaler.fit_transform(feature_matrix)

# Train Random Forest
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
rf_classifier.fit(feature_matrix_scaled, train_labels)

# Find best threshold on training data
train_similarities = train_df['basic_similarity'].values
thresholds = np.arange(0.4, 0.8, 0.02)
best_threshold = 0.6
best_acc = 0

for thresh in thresholds:
    # High similarity = Real = 0, Low similarity = Fake = 1
    preds = (train_similarities < thresh).astype(int)
    acc = accuracy_score(train_labels, preds)
    if acc > best_acc:
        best_acc = acc
        best_threshold = thresh

print(f"📈 TRAINING RESULTS:")
print(f"   Best threshold: {best_threshold:.3f} (training acc: {best_acc:.3f})")

# Process TEST data
print(f"\n🧪 PROCESSING TEST DATA...")
start_time = time.time()
test_features_data = []

print("Processing fake test articles...")
for idx, row in fake_test.iterrows():
    if idx % 50 == 0:
        print(f"  Fake test progress: {idx}/{TEST_SIZE_PER_CLASS}")
    features = calculate_features_fast(row['title'], row['text'])
    features.update({'label': 1, 'type': 'fake'})
    test_features_data.append(features)

print("Processing real test articles...")
for idx, row in real_test.iterrows():
    if idx % 50 == 0:
        print(f"  Real test progress: {idx}/{TEST_SIZE_PER_CLASS}")
    features = calculate_features_fast(row['title'], row['text'])
    features.update({'label': 0, 'type': 'real'})
    test_features_data.append(features)

testing_time = time.time() - start_time
print(f"✅ Testing completed in {testing_time:.1f} seconds")

# Test predictions
test_df = pd.DataFrame(test_features_data)
test_feature_matrix = test_df[feature_columns].values
test_labels = test_df['label'].values

print(f"\nTest data distribution:")
print(f"   Fake (label=1): {sum(test_labels == 1):,}")
print(f"   Real (label=0): {sum(test_labels == 0):,}")

# Scale test features
test_feature_matrix_scaled = scaler.transform(test_feature_matrix)

# Make predictions
rf_predictions = rf_classifier.predict(test_feature_matrix_scaled)
rf_probabilities = rf_classifier.predict_proba(test_feature_matrix_scaled)[:, 0]  # Prob of being real

# Simple threshold predictions
test_similarities = test_df['basic_similarity'].values
simple_predictions = (test_similarities < best_threshold).astype(int)

# Calculate accuracies
rf_accuracy = accuracy_score(test_labels, rf_predictions)
simple_accuracy = accuracy_score(test_labels, simple_predictions)

print(f"\n" + "="*80)
print("🎯 LARGE-SCALE RESULTS (MANAGEABLE SIZE)")
print("="*80)

print(f"📊 SCALE:")
print(f"   Training articles: {len(train_labels):,}")
print(f"   Test articles: {len(test_labels):,}")
print(f"   Processing time: {training_time + testing_time:.1f} seconds")

print(f"\n📈 PERFORMANCE:")
print(f"   Simple Threshold: {simple_accuracy:.3f} ({simple_accuracy*100:.1f}%)")
print(f"   Random Forest:    {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
print(f"   Improvement:      +{(rf_accuracy-simple_accuracy)*100:.1f} percentage points")

# Feature importance
feature_importance = rf_classifier.feature_importances_
print(f"\n🔥 FEATURE IMPORTANCE:")
feature_pairs = list(zip(feature_columns, feature_importance))
feature_pairs.sort(key=lambda x: x[1], reverse=True)
for feature, importance in feature_pairs:
    print(f"   {feature:20}: {importance:.3f}")

# Detailed classification report
print(f"\n📋 DETAILED RESULTS (Random Forest):")
print(classification_report(test_labels, rf_predictions, 
                          target_names=['Real News', 'Fake News'], digits=3))

# Feature differences
fake_test_features = test_df[test_df['label'] == 1]
real_test_features = test_df[test_df['label'] == 0]

print(f"\n📊 FEATURE PATTERNS:")
for feature in feature_columns:
    fake_avg = fake_test_features[feature].mean()
    real_avg = real_test_features[feature].mean()
    difference = abs(real_avg - fake_avg)
    print(f"   {feature:20}: Fake={fake_avg:.3f}, Real={real_avg:.3f}, Diff={difference:.3f}")

# Show some prediction examples
print(f"\n🔍 SAMPLE PREDICTIONS:")
sample_size = min(10, len(test_labels))
for i in range(sample_size):
    actual = "FAKE" if test_labels[i] == 1 else "REAL"
    rf_pred = "FAKE" if rf_predictions[i] == 1 else "REAL" 
    simple_pred = "FAKE" if simple_predictions[i] == 1 else "REAL"
    confidence = rf_probabilities[i]
    
    rf_status = "✅" if rf_pred == actual else "❌"
    simple_status = "✅" if simple_pred == actual else "❌"
    
    print(f"   {i+1:2d}. RF: {rf_status} {rf_pred} ({confidence:.2f}) | Simple: {simple_status} {simple_pred} | Actual: {actual}")

# ROC curves
fpr_rf, tpr_rf, _ = roc_curve(test_labels, rf_probabilities)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_simple, tpr_simple, _ = roc_curve(test_labels, test_similarities)
roc_auc_simple = auc(fpr_simple, tpr_simple)

print(f"\n📈 ROC AUC SCORES:")
print(f"   Simple Threshold: {roc_auc_simple:.3f}")
print(f"   Random Forest: {roc_auc_rf:.3f}")

# Final assessment
print(f"\n" + "="*80)
print("🏁 REALISTIC PERFORMANCE ASSESSMENT")
print("="*80)

print(f"✅ DATASET SCALE: Large enough for reliable results")
print(f"   {len(train_labels):,} training + {len(test_labels):,} test = {len(train_labels)+len(test_labels):,} total")

print(f"\n📊 REAL ACCURACY:")
if rf_accuracy > 0.85:
    print(f"   🎉 Excellent: {rf_accuracy*100:.1f}% (production-ready)")
elif rf_accuracy > 0.75:
    print(f"   ✅ Good: {rf_accuracy*100:.1f}% (usable with caution)")
elif rf_accuracy > 0.65:
    print(f"   ⚠️  Fair: {rf_accuracy*100:.1f}% (needs improvement)")
else:
    print(f"   ❌ Poor: {rf_accuracy*100:.1f}% (back to drawing board)")

if rf_accuracy - simple_accuracy > 0.1:
    print(f"   🚀 Random Forest provides significant improvement")
else:
    print(f"   🤔 Random Forest improvement is marginal")

# Create visualization
plt.figure(figsize=(12, 8))

# Accuracy comparison
plt.subplot(2, 3, 1)
methods = ['Simple\nThreshold', 'Random\nForest']
accuracies = [simple_accuracy, rf_accuracy]
colors = ['lightcoral', 'lightgreen']
bars = plt.bar(methods, accuracies, color=colors)
plt.ylabel('Accuracy')
plt.title(f'Performance Comparison\n({len(test_labels):,} test articles)')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.grid(True, alpha=0.3)

# ROC curves
plt.subplot(2, 3, 2)
plt.plot(fpr_simple, tpr_simple, 'r-', linewidth=2, label=f'Simple (AUC={roc_auc_simple:.3f})')
plt.plot(fpr_rf, tpr_rf, 'g-', linewidth=2, label=f'Random Forest (AUC={roc_auc_rf:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# Feature importance
plt.subplot(2, 3, 3)
features_short = [f.replace('_', '\n') for f in feature_columns]
plt.barh(range(len(feature_columns)), feature_importance)
plt.yticks(range(len(feature_columns)), features_short, fontsize=9)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.grid(True, alpha=0.3)

# Confusion matrix
plt.subplot(2, 3, 4)
rf_cm = confusion_matrix(test_labels, rf_predictions)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Random Forest\nConfusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Most important feature distribution
plt.subplot(2, 3, 5)
most_important_feature = feature_pairs[0][0]
fake_vals = fake_test_features[most_important_feature]
real_vals = real_test_features[most_important_feature]
plt.hist(fake_vals, bins=20, alpha=0.7, label='Fake', color='red', density=True)
plt.hist(real_vals, bins=20, alpha=0.7, label='Real', color='green', density=True)
plt.xlabel(most_important_feature.replace('_', ' ').title())
plt.ylabel('Density')
plt.title(f'Most Important Feature:\n{most_important_feature.replace("_", " ").title()}')
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy by confidence
plt.subplot(2, 3, 6)
confidence_bins = [0.5, 0.7, 0.8, 0.9, 1.0]
conf_accuracies = []
conf_counts = []

for i in range(len(confidence_bins)-1):
    low, high = confidence_bins[i], confidence_bins[i+1]
    mask = (rf_probabilities >= low) & (rf_probabilities < high)
    if np.sum(mask) > 10:  # Only if enough samples
        acc = accuracy_score(test_labels[mask], rf_predictions[mask])
        conf_accuracies.append(acc)
        conf_counts.append(np.sum(mask))
    else:
        conf_accuracies.append(0)
        conf_counts.append(0)

x_labels = [f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}' for i in range(len(confidence_bins)-1)]
bars = plt.bar(x_labels, conf_accuracies, color='skyblue')
plt.ylabel('Accuracy')
plt.xlabel('Confidence Range')
plt.title('Accuracy by Confidence Level')
plt.xticks(rotation=45)
for bar, count in zip(bars, conf_counts):
    if count > 0:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'n={count}', ha='center', va='bottom', fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'realistic_fnd_results_{len(test_labels)}_test.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved as 'realistic_fnd_results_{len(test_labels)}_test.png'")
plt.show()

# Save results
results = {
    'dataset_info': {
        'original_fake_count': len(df_fake),
        'original_real_count': len(df_real), 
        'train_per_class': TRAIN_SIZE_PER_CLASS,
        'test_per_class': TEST_SIZE_PER_CLASS
    },
    'performance': {
        'simple_threshold': {
            'threshold': float(best_threshold),
            'accuracy': float(simple_accuracy),
            'roc_auc': float(roc_auc_simple)
        },
        'random_forest': {
            'accuracy': float(rf_accuracy),
            'roc_auc': float(roc_auc_rf)
        }
    },
    'feature_importance': {feature: float(importance) 
                         for feature, importance in zip(feature_columns, feature_importance)},
    'processing_time_seconds': training_time + testing_time
}

with open('realistic_scale_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n🎉 ANALYSIS COMPLETE!")
print(f"💾 Results saved to 'realistic_scale_results.json'")
print(f"⏱️  Total processing time: {training_time + testing_time:.1f} seconds")

# Final honest assessment
print(f"\n🎯 HONEST PERFORMANCE REPORT:")
print(f"   Training size: {TRAIN_SIZE_PER_CLASS*2:,} articles (good for ML)")
print(f"   Test size: {TEST_SIZE_PER_CLASS*2:,} articles (reliable evaluation)")
print(f"   Random Forest accuracy: {rf_accuracy*100:.1f}%")

if rf_accuracy > 0.8:
    print(f"   ✅ This is likely REAL performance you can trust")
elif rf_accuracy > 0.7:
    print(f"   ✅ Good performance - usable in practice")
else:
    print(f"   ⚠️  Needs more work - consider more features or data")

print(f"\n🚀 RECOMMENDATION:")
if rf_accuracy > simple_accuracy + 0.05:
    print(f"   ✅ Use Random Forest - clear improvement over simple threshold")
else:
    print(f"   🤔 Simple threshold might be sufficient - Random Forest not much better")