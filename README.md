# Fake News Detection Using Multi-Agent LLM Ensemble

A multi-agent system that uses an LLM (Qwen 7B) as 4 specialized "agents" to detect fake news. Each agent scores a news article on a different dimension, and a neural network learns the optimal combination of these scores to classify articles as REAL or FAKE.

## Results

| Config | Train | Test | Accuracy | Precision | Recall | F1 | ROC AUC |
|--------|-------|------|----------|-----------|--------|-----|---------|
| Small  | 400   | 100  | **96%**  | 0.96      | 0.96   | 0.96| 0.99    |
| Medium | 640   | 200  | **94%**  | 0.92      | 0.96   | 0.94| 0.98    |

Dataset: ISOT Fake News Dataset (Ahmed et al., 2017) — Real articles from Reuters, Fake articles flagged by Politifact.

---

## How It Works (The Big Picture)

```
News Article
     |
     v
+------------------+
| 4 LLM Agents     |  Each agent reads the article and gives a score (0 to 1)
|                  |
| 1. Style Agent   |  → Checks source credibility (Reuters dateline? Named sources?)
| 2. Sentiment     |  → Checks emotional manipulation (fear-mongering? outrage bait?)
| 3. Vocab Agent   |  → Checks sensationalist language (ALL CAPS? clickbait?)
| 4. Semantic      |  → Checks factual coherence (supported claims? logical?)
+------------------+
     |
     v  [4 scores: e.g., 0.1, 0.2, 0.1, 0.15]
     |
+------------------+
| Ensemble Layer   |  Neural network (4 → 8 → 1) learns optimal combination
| (trained weights)|  Outputs a single probability
+------------------+
     |
     v
  REAL or FAKE  (based on learned threshold)
```

---

## The 4 Parameters (Agents)

These are the 4 dimensions along which we analyze each news article. Each agent is the SAME LLM (Qwen 7B) but with a different prompt.

### 1. Style Agent — Source Credibility
**What it checks:** Does the article cite named sources? Does it have a dateline (e.g., "WASHINGTON (Reuters)")? Are there official quotes, institutional references, reporter bylines?

- Score 0.0 = Strong credibility markers → likely REAL
- Score 1.0 = No sources, anonymous claims → likely FAKE

### 2. Sentiment Agent — Emotional Manipulation
**What it checks:** Does the article use fear-mongering, outrage bait, us-vs-them framing, conspiracy language, emotional appeals over facts?

- Score 0.0 = Neutral, factual reporting → likely REAL
- Score 1.0 = Heavy emotional manipulation → likely FAKE

### 3. Vocab Agent — Sensationalist Language
**What it checks:** Does the article use ALL CAPS, excessive exclamation marks, clickbait phrases ("SHOCKING", "You won't believe"), loaded/inflammatory words?

- Score 0.0 = Measured, journalistic language → likely REAL
- Score 1.0 = Inflammatory, clickbait language → likely FAKE

### 4. Semantic Agent — Factual Coherence
**What it checks:** Are the claims verifiable and logically consistent? Are there extraordinary unsupported claims, conspiracy theories, or internal contradictions?

- Score 0.0 = Well-supported, verifiable claims → likely REAL
- Score 1.0 = Unsupported claims, contradictions → likely FAKE

---

## Architecture Deep Dive

### Step 1: LLM Scoring (the expensive part)
- We load **Qwen/Qwen2.5-7B-Instruct** with 4-bit quantization (fits on a T4 GPU)
- For each article, we run it through all 4 agents (4 different prompts)
- Each agent outputs a JSON like `{"score": 0.25}`
- The model loads ONCE and processes all train/val/test splits, then unloads
- Temperature = 0 (greedy decoding) for fully deterministic, reproducible scores

### Step 2: StandardScaler
- Agent scores are standardized (zero mean, unit variance) using sklearn's StandardScaler
- The scaler is fitted ONLY on training data (no data leakage)
- Saved to `scaler.gz` for reproducibility

### Step 3: Ensemble Layer (the learning part)
```python
nn.Sequential(
    nn.Linear(4, 8),   # 4 agent scores → 8 hidden neurons
    nn.ReLU(),          # Non-linear activation
    nn.Linear(8, 1)     # 8 → 1 output (fake probability)
)
```
- This is a tiny neural network (41 parameters total)
- Trained with BCE loss (Binary Cross-Entropy with Logits)
- Optimizer: Adam
- The ensemble learns which agents matter most and how to combine them non-linearly

### Step 4: Threshold Optimization
- After training, we find the optimal classification threshold using the ROC curve on the VALIDATION set
- We use Youden's J statistic (maximizes TPR - FPR)
- This threshold is saved alongside the weights

### Step 5: Evaluation
- The held-out TEST set (never seen during training or threshold selection) is evaluated
- Metrics: Accuracy, Precision, Recall, F1-Score

---

## Why These Prompts Work

The key insight: **generic writing quality doesn't separate fake from real news.** Both can be well-written or poorly written.

What DOES work is checking for **specific fake news signals**:

| Old Prompt (didn't work) | New Prompt (works) | Why |
|--------------------------|-------------------|-----|
| "Rate professionalism" | "Check for source credibility markers" | Reuters articles have datelines, fake ones don't |
| "Rate emotional charge" | "Check for manipulation tactics" | Real news reports emotion, fake news weaponizes it |
| "Rate vocabulary complexity" | "Check for sensationalist language" | Complexity isn't fake; clickbait IS |
| "Rate semantic clarity" | "Check for factual coherence" | Clarity isn't fake; unsupported claims ARE |

This change alone took accuracy from **49% → 94%**.

---

## Project Structure

```
FND/
├── new_test.py              # Main file — everything is here
├── README.md                # This file
└── Dataset/
    ├── True.csv             # Full ISOT real news (21k articles, ~53MB)
    ├── Fake.csv             # Full ISOT fake news (23k articles, ~62MB)
    ├── True_small.csv       # 500 articles subset (1.2MB) — for 96% config
    ├── Fake_small.csv       # 500 articles subset (1.2MB) — for 96% config
    ├── True_medium.csv      # 800 articles subset (1.9MB) — for 94% config
    └── Fake_medium.csv      # 800 articles subset (2.0MB) — for 94% config
```

---

## How to Run

### Prerequisites
- Google Colab account (free tier works)
- T4 GPU runtime

### Steps on Google Colab

**Cell 1 — Install dependencies:**
```python
!pip install transformers accelerate bitsandbytes scikit-learn matplotlib joblib -q
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
```

**Cell 2 — Upload files:**
```python
import os
os.makedirs('./Dataset', exist_ok=True)
from google.colab import files

print("Upload new_test.py:")
uploaded = files.upload()

print("\nUpload True_small.csv:")
uploaded = files.upload()
for f in uploaded: os.rename(f, './Dataset/True.csv')

print("\nUpload Fake_small.csv:")
uploaded = files.upload()
for f in uploaded: os.rename(f, './Dataset/Fake.csv')
```

**Cell 3 — Run:**
```python
!pip install -U bitsandbytes -q
!python new_test.py
```

**Cell 4 — View plots:**
```python
from IPython.display import Image, display
display(Image('loss_vs_epoch.png'))
display(Image('roc_curve.png'))
```

### Runtime
- ~14 minutes on T4 GPU (small config)
- ~25 minutes on T4 GPU (medium config)

---

## Config Options

Located at the bottom of `new_test.py`:

```python
CONFIG = {
    "batch_size": 10,        # Articles per LLM inference batch
    "epochs": 200,           # Training epochs for ensemble layer
    "learning_rate": 0.005,  # Adam optimizer learning rate
    "patience": 20,          # Early stopping patience
    "model_name": "Qwen/Qwen2.5-7B-Instruct",  # LLM model
    "max_text_length": 1500, # Max characters per article sent to LLM
    "weights_save_path": "best_ensemble_weights.pth",
    "train_rows": 800,       # Total rows to use (split equally fake/real)
    "test_rows": 200,        # Held-out test size
    "val_split": 0.2,        # Fraction of train used for validation
}
```

### Configs that produced our results:

**96% Accuracy (small):**
```
epochs=50, lr=0.01, patience=10, train_rows=500, test_rows=100
```

**94% Accuracy (medium — more reliable):**
```
epochs=200, lr=0.005, patience=20, train_rows=800, test_rows=200
```

---

## What's NOT Random / Static

Everything is dynamically computed:

| Component | How |
|-----------|-----|
| Agent scores | Computed by LLM per article (deterministic, temperature=0) |
| Ensemble weights | Learned via backpropagation |
| Threshold | Computed via ROC curve on validation set |
| Scaler | Fitted on training data only |
| Train/val/test split | Stratified with random_state=42 (reproducible) |

No hardcoded weights. No random generation. Same input = same output every time.

---

## Evolution of the Project

1. **v1 — Hardcoded weights:** Manual weights like `style=0.49, vocab=0.22`. Accuracy: unmeasured
2. **v2 — Random weights:** Generated random weight combos and tested. Accuracy: ~50%
3. **v3 — Learned weights (linear):** `nn.Linear(4,1)` with generic prompts. Accuracy: **49%** (random chance — prompts were wrong)
4. **v4 — Rewritten prompts:** Targeted fake news signals instead of generic quality. Accuracy: **94%**
5. **v5 — Deeper ensemble:** `4→8→1` with ReLU, more epochs. Accuracy: **96%** (100 test) / **94%** (200 test)
