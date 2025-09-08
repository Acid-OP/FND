Let me explain the complete flow with concrete examples:
Step 1: Individual Agent Scores
Let's say we have 2 articles in a batch:
python# Article 1: "Scientists discover new planet..."
# Article 2: "SHOCKING! Government LIES exposed!!!"

# Each agent gives scores
individual_scores = {
    'style':     [0.2, 0.9],  # Article 1: professional, Article 2: unprofessional
    'vocab':     [0.1, 0.8],  # Article 1: good vocab, Article 2: poor vocab  
    'sentiment': [0.0, 0.95], # Article 1: neutral, Article 2: very emotional
    'semantics': [0.3, 0.7]   # Article 1: logical, Article 2: questionable
}

# Agent weights:
weights = {
    'style': 0.3,      # 30%
    'vocab': 0.2,      # 20% 
    'sentiment': 0.25, # 25%
    'semantics': 0.25  # 25%
}
Step 2: Calculate Weighted Scores
For each article, calculate: weighted_score = Σ(agent_score × agent_weight)
python# Article 1 weighted score:
weighted_score_1 = (0.2 × 0.3) + (0.1 × 0.2) + (0.0 × 0.25) + (0.3 × 0.25)
weighted_score_1 = 0.06 + 0.02 + 0.0 + 0.075 = 0.155

# Article 2 weighted score:  
weighted_score_2 = (0.9 × 0.3) + (0.8 × 0.2) + (0.95 × 0.25) + (0.7 × 0.25)
weighted_score_2 = 0.27 + 0.16 + 0.2375 + 0.175 = 0.8425

# Result:
all_weighted_scores = [0.155, 0.8425]
all_labels = [0, 1]  # 0=REAL, 1=FAKE
Step 3: Complete Dataset Processing
After processing ALL batches (fake + real):
python# Example final data:
all_weighted_scores = [0.155, 0.234, 0.189, 0.298, 0.167,  # 5 REAL articles (labels = 0)
                       0.842, 0.756, 0.923, 0.678, 0.889]  # 5 FAKE articles (labels = 1)

all_labels =         [0,     0,     0,     0,     0,        # REAL
                      1,     1,     1,     1,     1]        # FAKE
Step 4: ROC Curve & Threshold Calculation
pythonfrom sklearn.metrics import roc_curve

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_weighted_scores)

# Example values:
# thresholds = [1.923, 0.923, 0.889, 0.842, 0.756, 0.678, 0.298, 0.234, 0.189, 0.167, 0.155]
# tpr =        [0.0,   0.2,   0.4,   0.6,   0.8,   1.0,   1.0,   1.0,   1.0,   1.0,   1.0]
# fpr =        [0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.2,   0.4,   0.6,   0.8,   1.0]

# Find optimal threshold (maximize TPR - FPR)
threshold_idx = np.argmax(tpr - fpr)  # Let's say index = 5
THRESHOLD = thresholds[threshold_idx]  # THRESHOLD = 0.678
Step 5: Making Predictions with Threshold
pythonTHRESHOLD = 0.678

# For each article:
# If weighted_score >= THRESHOLD → Predict FAKE
# If weighted_score < THRESHOLD → Predict REAL

predictions = []
for score in all_weighted_scores:
    if score >= THRESHOLD:
        predictions.append(1)  # FAKE
    else:
        predictions.append(0)  # REAL

# Results:
# all_weighted_scores = [0.155, 0.234, 0.189, 0.298, 0.167, 0.842, 0.756, 0.923, 0.678, 0.889]
# predictions =         [0,     0,     0,     0,     0,     1,     1,     1,     1,     1]
# actual_labels =       [0,     0,     0,     0,     0,     1,     1,     1,     1,     1]
#                       ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓      ✓
Step 6: Calculate Accuracy
python# Count correct predictions:
fake_correct = 0  # Correctly identified fake articles
fake_wrong = 0    # Incorrectly identified fake articles  
real_correct = 0  # Correctly identified real articles
real_wrong = 0    # Incorrectly identified real articles

# Check fake articles (last 5 scores)
for score in all_weighted_scores[5:]:  # [0.842, 0.756, 0.923, 0.678, 0.889]
    if score >= THRESHOLD:  # Should be >= for fake
        fake_correct += 1   # All 5 are >= 0.678, so fake_correct = 5
    else:
        fake_wrong += 1     # fake_wrong = 0

# Check real articles (first 5 scores)  
for score in all_weighted_scores[:5]:  # [0.155, 0.234, 0.189, 0.298, 0.167]
    if score < THRESHOLD:   # Should be < for real
        real_correct += 1   # All 5 are < 0.678, so real_correct = 5
    else:
        real_wrong += 1     # real_wrong = 0

# Final results:
total_correct = fake_correct + real_correct  # 5 + 5 = 10
accuracy = (total_correct / 10) * 100        # 100%