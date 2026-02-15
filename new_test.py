# =====================================================================================
# FAKE NEWS DETECTION USING MULTI-AGENT LLM ENSEMBLE
# =====================================================================================
#
# This system uses a single LLM (Qwen 7B) prompted as 4 different "agents", each
# analyzing a news article from a different angle (style, sentiment, vocab, semantics).
# Each agent outputs a score between 0 (likely REAL) and 1 (likely FAKE).
# A small neural network (ensemble layer) then learns the optimal way to combine
# these 4 scores into a final REAL/FAKE prediction.
#
# Architecture:
#   Article → [Style=0.1, Sentiment=0.2, Vocab=0.1, Semantic=0.15] → Ensemble(4→8→1) → FAKE/REAL
#
# Results:
#   96% accuracy on 100 test samples (ISOT Fake News Dataset)
#   94% accuracy on 200 test samples (more reliable)
# =====================================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import numpy as np
import gc
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# =====================================================================================
# 1. DATASET SETUP
# =====================================================================================
class NewsDataset(Dataset):
    """
    Standard PyTorch Dataset wrapper for a pandas DataFrame.
    Each item returns (article_text, label) where label is 0=REAL, 1=FAKE.
    """
    def __init__(self, dataframe):
        # reset_index ensures iloc indexing works correctly after train/val/test splits
        self.data_frame = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        text = row['text']
        label = row['label']
        return text, label

# =====================================================================================
# 2. LLM MANAGER & AGENT BLUEPRINT
# =====================================================================================
class BaseAgentClassifier:
    """
    Manages a single shared LLM instance used by all 4 agents.

    The model is loaded once into GPU memory with 4-bit quantization (to fit on
    a free Colab T4 with 15GB VRAM), shared across all agents for inference,
    then unloaded to free memory.

    Key design choices:
    - 4-bit quantization (NF4): Reduces 7B model from ~14GB to ~4GB VRAM
    - Temperature = 0: Greedy decoding for deterministic, reproducible scores
    - max_new_tokens = 30: We only need {"score": 0.XX} (~10 tokens)
    - Stop at "}": Terminates generation as soon as JSON is complete
    """
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = None       # Model loaded on-demand, not at init
        self.tokenizer = None

    def _load_model(self):
        """Loads the LLM and tokenizer into GPU memory with 4-bit quantization."""
        if self.model is not None:
            return  # Already loaded, skip

        print(f"\nLoading {self.model_name} onto {self.device}...")

        # 4-bit quantization config — makes 7B model fit in ~4GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,              # Use 4-bit precision
            bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 for speed
            bnb_4bit_quant_type="nf4",      # NormalFloat4 quantization (best quality)
            bnb_4bit_use_double_quant=True,  # Quantize the quantization constants too
        )

        model_kwargs = {
            'low_cpu_mem_usage': True,       # Don't duplicate model in CPU RAM
            'quantization_config': bnb_config,
            'device_map': 'auto',            # Let HuggingFace decide GPU/CPU split
            'max_memory': {0: "10GB", "cpu": "30GB"}  # Memory limits per device
        }

        # Load tokenizer with left padding (needed for batched generation)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the actual model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        print("Model loaded successfully.")

    def _unload_model(self):
        """Unloads the model from memory and clears GPU cache."""
        if self.model is not None:
            print(f"\nUnloading {self.model_name}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Free GPU VRAM
            gc.collect()                  # Free CPU RAM
            print("Model unloaded and memory cleared.")

    def _extract_score(self, response_text):
        """
        Extracts a float score (0.0 to 1.0) from the LLM's response.

        Strategy:
        1. First tries to parse a JSON object like {"score": 0.25}
        2. Falls back to regex matching any decimal number between 0 and 1
        3. Returns None if both fail (caller defaults to 0.5)
        """
        # Strategy 1: Try parsing JSON (the expected output format)
        try:
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                score = float(data['score'])
                if 0 <= score <= 1:
                    return score
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        # Strategy 2: Regex fallback — find any decimal like "0.75" in the text
        matches = re.findall(r'(?:^|[\s:])([01]\.\d{1,2})\b', response_text)
        for m in matches:
            try:
                score = float(m)
                if 0 <= score <= 1:
                    return score
            except ValueError:
                continue

        return None  # Both strategies failed

    def run_inference(self, prompt_dicts, agent_name, max_new_tokens=30, temperature=0.0):
        """
        Runs batched LLM inference for a list of prompts.

        Args:
            prompt_dicts: List of dicts with 'full_prompt' key
            agent_name: Name of the agent (for logging)
            max_new_tokens: Max tokens to generate (30 is enough for {"score": 0.XX})
            temperature: 0.0 = greedy decoding (deterministic, reproducible)

        Returns:
            List of dicts with 'score' (float) and 'analysis' (raw text)
        """
        prompt_strings = [p['full_prompt'] for p in prompt_dicts]

        # Tokenize all prompts in the batch at once
        inputs = self.tokenizer(prompt_strings, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Stop generation when we see "}" (end of JSON) or end-of-sequence
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("}")
        ]

        results = []

        # Generate responses for all prompts in the batch (no gradient needed)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,  # Greedy when temp=0
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Extract scores from each generated response
        for i in range(len(outputs)):
            # Only decode the NEW tokens (skip the input prompt tokens)
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = outputs[i][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Ensure JSON is closed (sometimes generation stops mid-token)
            if not response_text.strip().endswith('}'):
                response_text += '}'

            # Extract the numerical score from the response
            score = self._extract_score(response_text)

            # Default to 0.5 (neutral) if extraction failed
            if score is None:
                score = 0.5

            results.append({
                'score': score,
                'analysis': response_text,
            })
        return results

# =====================================================================================
# 3. SPECIALIST AGENT DEFINITIONS
# =====================================================================================
#
# Each agent is the SAME LLM (Qwen 7B) but with a DIFFERENT prompt.
# The prompts are designed to detect specific fake news signals, not generic
# writing quality. This is the key insight that took accuracy from 49% to 94%.
#
# All agents output a score: 0.0 = likely REAL, 1.0 = likely FAKE
# =====================================================================================

def create_chat_prompt(system_prompt, user_prompt):
    """
    Creates a structured chat prompt using Qwen's special tokens.
    Format: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
    """
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

# Shared system prompt — tells the LLM to output only JSON
SYSTEM_PROMPT = """You are an expert fake news detector. Analyze the article and output ONLY a valid JSON object.
Example output: {"score": 0.25}"""


class NewsStyleClassifier:
    """
    AGENT 1: Source Credibility Analysis

    Checks if the article has markers of legitimate journalism:
    - Named sources and official citations
    - News agency attribution (Reuters, AP)
    - Datelines (e.g., "WASHINGTON (Reuters) -")
    - Reporter bylines and institutional references

    Real news from Reuters always has datelines and named sources.
    Fake news typically lacks attribution and uses anonymous claims.
    """
    def __init__(self):
        self.agent_name = "Style Agent"

    def _create_prompts(self, news_list, max_text_len):
        user_template = """Analyze this article for SOURCE CREDIBILITY signals.

Check for: named sources, official citations, news agency attribution (Reuters, AP), datelines, reporter bylines, institutional references.

0.0 = strong credibility markers (named sources, datelines, official quotes) → likely REAL
1.0 = no credibility markers (anonymous claims, no sources, no attribution) → likely FAKE

Example: "WASHINGTON (Reuters) - The Federal Reserve announced..." → {{"score": 0.10}}
Example: "Someone revealed the shocking truth that they don't want you to know..." → {{"score": 0.90}}

Article: {news}

JSON:"""
        prompts = []
        for news in news_list:
            user_prompt = user_template.format(news=news[:max_text_len])
            full_prompt = create_chat_prompt(SYSTEM_PROMPT, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts


class NewsSentimentClassifier:
    """
    AGENT 2: Emotional Manipulation Detection

    Checks if the article uses propaganda and manipulation tactics:
    - Fear-mongering and outrage bait
    - Us-vs-them framing
    - Conspiracy language
    - Call-to-action pressure
    - Emotional appeals over factual evidence

    Real news reports facts neutrally. Fake news weaponizes emotion.
    """
    def __init__(self):
        self.agent_name = "Sentiment Agent"

    def _create_prompts(self, news_list, max_text_len):
        user_template = """Analyze this article for EMOTIONAL MANIPULATION tactics.

Check for: fear-mongering, outrage bait, us-vs-them framing, conspiracy language, call-to-action pressure, emotional appeals over facts.

0.0 = neutral factual reporting, balanced tone → likely REAL
1.0 = heavy emotional manipulation, propaganda tactics → likely FAKE

Example: "The committee voted 12-8 to approve the measure after three hours of debate." → {{"score": 0.05}}
Example: "They are DESTROYING our country! Wake up people before it's too late!" → {{"score": 0.95}}

Article: {news}

JSON:"""
        prompts = []
        for news in news_list:
            user_prompt = user_template.format(news=news[:max_text_len])
            full_prompt = create_chat_prompt(SYSTEM_PROMPT, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts


class NewsVocabClassifier:
    """
    AGENT 3: Sensationalist Language Detection

    Checks if the article uses language patterns common in fake news:
    - ALL CAPS words for emphasis
    - Excessive exclamation marks
    - Clickbait phrases ("SHOCKING", "BREAKING", "You won't believe")
    - Loaded/inflammatory vocabulary
    - Hyperbolic claims and slang

    Real journalism uses measured, precise language.
    Fake news uses inflammatory, attention-grabbing language.
    """
    def __init__(self):
        self.agent_name = "Vocab Agent"

    def _create_prompts(self, news_list, max_text_len):
        user_template = """Analyze this article for SENSATIONALIST LANGUAGE patterns.

Check for: ALL CAPS words, excessive exclamation marks, clickbait phrases ("SHOCKING", "BREAKING", "You won't believe"), loaded/inflammatory words, hyperbolic claims, slang, informal language.

0.0 = measured, precise journalistic language → likely REAL
1.0 = inflammatory, sensationalist, clickbait language → likely FAKE

Example: "Officials reported a 3% increase in quarterly earnings." → {{"score": 0.05}}
Example: "EXPOSED! The SHOCKING scandal they tried to COVER UP!!!" → {{"score": 0.95}}

Article: {news}

JSON:"""
        prompts = []
        for news in news_list:
            user_prompt = user_template.format(news=news[:max_text_len])
            full_prompt = create_chat_prompt(SYSTEM_PROMPT, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts


class NewsSemanticClassifier:
    """
    AGENT 4: Factual Coherence Analysis

    Checks if the article's claims are plausible and internally consistent:
    - Are claims verifiable and evidence-based?
    - Is the narrative logically consistent?
    - Are there extraordinary unsupported claims?
    - Are there conspiracy theories or misrepresentation of facts?

    Real news makes verifiable claims. Fake news makes extraordinary
    unsupported claims and relies on conspiracy theories.
    """
    def __init__(self):
        self.agent_name = "Semantic Agent"

    def _create_prompts(self, news_list, max_text_len):
        user_template = """Analyze this article for FACTUAL COHERENCE and plausibility.

Check for: verifiable claims, logical consistency, extraordinary unsupported claims, internal contradictions, implausible narratives, conspiracy theories, misrepresentation of facts.

0.0 = well-supported, verifiable, logically consistent claims → likely REAL
1.0 = extraordinary unsupported claims, conspiracy theories, logical contradictions → likely FAKE

Example: "NASA confirmed the launch date after a review by the safety board." → {{"score": 0.10}}
Example: "Secret documents prove that the government is hiding aliens in Area 51." → {{"score": 0.90}}

Article: {news}

JSON:"""
        prompts = []
        for news in news_list:
            user_prompt = user_template.format(news=news[:max_text_len])
            full_prompt = create_chat_prompt(SYSTEM_PROMPT, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts

# =====================================================================================
# 4. ENSEMBLE LAYER
# =====================================================================================
class EnsembleLayer(nn.Module):
    """
    A small neural network that learns how to combine the 4 agent scores.

    Architecture: 4 → 8 → 1

        4 input scores          8 hidden neurons              1 output
        [style]    ──┐       ┌──[h1]──┐
        [sentiment]──┼───────┤  [h2]  ├──── [FAKE probability]
        [vocab]    ──┤       │  ...   │
        [semantic] ──┘       └──[h8]──┘

        Layer 1: Linear(4,8)   ReLU    Layer 2: Linear(8,1)

    Why not just a weighted average (Linear 4→1)?
    - A weighted average can only do: w1*style + w2*sentiment + w3*vocab + w4*semantic
    - This hidden layer can learn INTERACTIONS between agents, like:
      "if style is low AND sentiment is high → very likely fake"
    - The ReLU activation enables these non-linear combinations

    Total parameters: 41 (4*8 + 8 bias + 8*1 + 1 bias)
    Trains in milliseconds — all the time is spent on LLM inference.

    Adding this hidden layer improved accuracy from 94% → 96%.
    """
    def __init__(self, num_agents):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_agents, 8),  # 4 agent scores → 8 hidden neurons
            nn.ReLU(),                 # Non-linear activation (enables interaction learning)
            nn.Linear(8, 1)            # 8 hidden → 1 output (fake probability logit)
        )

    def forward(self, x):
        # x shape: [batch_size, 4] → output shape: [batch_size, 1]
        return self.net(x)

# =====================================================================================
# 5. ORCHESTRATOR CLASS
# =====================================================================================
class AggregatedNewsClassifier:
    """
    The main orchestrator that ties everything together:
    1. Manages the LLM and 4 agents
    2. Collects scores from all agents for all articles
    3. Trains the ensemble layer to learn optimal score combination
    4. Finds the optimal classification threshold
    5. Evaluates on held-out test data
    """
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Single shared LLM instance — all 4 agents use the same model
        self.llm_manager = BaseAgentClassifier(model_name=config["model_name"])

        # The 4 specialist agents (each is just a different prompt template)
        self.agents = {
            "style": NewsStyleClassifier(), "sentiment": NewsSentimentClassifier(),
            "vocab": NewsVocabClassifier(), "semantic": NewsSemanticClassifier()
        }
        self.agent_order = ["style", "sentiment", "vocab", "semantic"]

        # The learnable ensemble layer (4→8→1 neural network)
        self.ensemble_layer = EnsembleLayer(num_agents=len(self.agents)).to(self.device)

        # Binary Cross-Entropy loss with logits (sigmoid is applied internally)
        self.criterion = nn.BCEWithLogitsLoss()

        # Classification threshold — will be learned from validation data
        self.optimal_threshold = 0.5

        # StandardScaler — normalizes agent scores to zero mean, unit variance
        # This prevents any single agent from dominating due to scale differences
        self.scaler = StandardScaler()

    def _collect_scores_from_loader(self, dataloader, split_name):
        """
        Runs all 4 agents on every article in a dataloader.
        The LLM must already be loaded before calling this.

        For each batch of articles:
          1. Run Style Agent → get style scores
          2. Run Sentiment Agent → get sentiment scores
          3. Run Vocab Agent → get vocab scores
          4. Run Semantic Agent → get semantic scores
          5. Stack into [batch_size, 4] matrix

        Returns:
            all_scores: numpy array of shape [num_samples, 4]
            all_labels: numpy array of shape [num_samples]
        """
        all_scores_list = []
        all_labels_list = []

        for batch_idx, (news, labels) in enumerate(dataloader):
            print(f"  [{split_name}] Batch {batch_idx + 1}/{len(dataloader)}")
            batch_scores = {}

            # Run each agent on the same batch of articles
            for agent_key in self.agent_order:
                agent = self.agents[agent_key]
                prompt_dicts = agent._create_prompts(news, self.config["max_text_length"])
                results = self.llm_manager.run_inference(prompt_dicts, agent.agent_name)
                batch_scores[agent_key] = np.array([res['score'] for res in results])

            # Stack 4 score arrays into a [batch_size, 4] matrix
            scores_stacked = np.stack([batch_scores[key] for key in self.agent_order], axis=1)
            all_scores_list.append(scores_stacked)
            all_labels_list.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

        # Concatenate all batches into final arrays
        all_scores = np.vstack(all_scores_list)
        all_labels = np.concatenate(all_labels_list)

        # Print diagnostic stats — helps verify agents are working correctly
        print(f"\n  [{split_name}] Collected {len(all_labels)} samples.")
        print(f"  Score stats per agent:")
        for i, name in enumerate(self.agent_order):
            col = all_scores[:, i]
            default_count = np.sum(col == 0.5)  # Count failed extractions (defaulted to 0.5)
            print(f"    {name}: mean={col.mean():.3f}, std={col.std():.3f}, min={col.min():.3f}, max={col.max():.3f}, defaulted_to_0.5={default_count}/{len(col)}")
        print(f"  Labels: fake={np.sum(all_labels==1)}, real={np.sum(all_labels==0)}")
        return all_scores, all_labels

    def collect_all_scores(self, train_loader, val_loader, test_loader):
        """
        Loads the LLM ONCE, scores all 3 data splits, then unloads.

        This is the most expensive step (~14-25 min on T4 GPU).
        Loading the model once instead of 3 times saves ~4 minutes.
        """
        print(f"\n{'='*60}\nCOLLECTING ALL AGENT SCORES (model loads once)\n{'='*60}\n")

        self.llm_manager._load_model()

        train_scores, train_labels = self._collect_scores_from_loader(train_loader, "TRAIN")
        val_scores, val_labels = self._collect_scores_from_loader(val_loader, "VAL")
        test_scores, test_labels = self._collect_scores_from_loader(test_loader, "TEST")

        self.llm_manager._unload_model()

        return (train_scores, train_labels), (val_scores, val_labels), (test_scores, test_labels)

    def train_weights(self, train_scores, train_labels, val_scores, val_labels):
        """
        Trains the ensemble layer on pre-computed agent scores.

        This is FAST (milliseconds) because:
        - No LLM inference needed (scores are already cached as numpy arrays)
        - The ensemble layer has only 41 trainable parameters
        - We just do matrix multiplication + backprop on small tensors

        Features:
        - StandardScaler: normalizes scores so no agent dominates
        - Early stopping: stops if validation loss doesn't improve for `patience` epochs
        - Best model saved: always keeps the best weights based on val loss
        """
        optimizer = torch.optim.Adam(self.ensemble_layer.parameters(), lr=self.config["learning_rate"])
        best_loss = float('inf')
        patience_counter = 0
        epoch_losses = []

        # Fit scaler on TRAINING data only (no data leakage from val/test)
        self.scaler.fit(train_scores)
        joblib.dump(self.scaler, 'scaler.gz')
        print("Scaler fitted on training data and saved to 'scaler.gz'.")

        # Standardize scores and convert to PyTorch tensors
        scaled_train = self.scaler.transform(train_scores)
        train_scores_t = torch.FloatTensor(scaled_train).to(self.device)
        train_labels_t = torch.FloatTensor(train_labels).view(-1, 1).to(self.device)

        scaled_val = self.scaler.transform(val_scores)
        val_scores_t = torch.FloatTensor(scaled_val).to(self.device)
        val_labels_t = torch.FloatTensor(val_labels).view(-1, 1).to(self.device)

        print(f"\n{'='*60}\nSTARTING WEIGHT TRAINING\n{'='*60}")
        print(f"Epochs: {self.config['epochs']}, LR: {self.config['learning_rate']}, Early stop patience: {self.config['patience']}")
        print(f"Train samples: {len(train_labels)}, Val samples: {len(val_labels)}\n")

        for epoch in range(self.config["epochs"]):
            # --- Forward pass on training data ---
            self.ensemble_layer.train()
            logits = self.ensemble_layer(train_scores_t)       # [batch, 1]
            loss = self.criterion(logits, train_labels_t)       # BCE loss

            # --- Backpropagation ---
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute new gradients
            optimizer.step()       # Update the 41 parameters

            # --- Evaluate on validation data (no gradient needed) ---
            self.ensemble_layer.eval()
            with torch.no_grad():
                val_logits = self.ensemble_layer(val_scores_t)
                val_loss = self.criterion(val_logits, val_labels_t).item()

            epoch_losses.append(val_loss)
            print(f"Epoch {epoch + 1:3d}/{self.config['epochs']}  |  Train Loss: {loss.item():.4f}  |  Val Loss: {val_loss:.4f}", end="")

            # --- Save best model + early stopping ---
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.ensemble_layer.state_dict(), self.config["weights_save_path"])
                print(f"  << saved (best)")
            else:
                patience_counter += 1
                print(f"  (no improve {patience_counter}/{self.config['patience']})")
                if patience_counter >= self.config["patience"]:
                    print(f"\nEarly stopping at epoch {epoch + 1}.")
                    break

        # Reload the best weights (not necessarily the last epoch's weights)
        self.ensemble_layer.load_state_dict(torch.load(self.config["weights_save_path"], weights_only=True))

        self.plot_training_loss(epoch_losses)
        self.find_optimal_threshold(val_scores, val_labels)

    def find_optimal_threshold(self, val_scores, val_labels):
        """
        Finds the best classification threshold using the ROC curve.

        Instead of using a fixed threshold of 0.5, we find the threshold that
        maximizes Youden's J statistic (TPR - FPR) on the VALIDATION set.

        This is important because the ensemble output distribution may not be
        centered at 0.5. The optimal threshold is learned from data.
        """
        print("\nFinding optimal threshold on validation data...")
        self.ensemble_layer.eval()

        # Get predictions on validation set
        scaled = self.scaler.transform(val_scores)
        scores_t = torch.FloatTensor(scaled).to(self.device)

        with torch.no_grad():
            logits = self.ensemble_layer(scores_t)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()  # Convert logits → probabilities

        # Compute ROC curve and find optimal threshold
        fpr, tpr, thresholds = roc_curve(val_labels, preds)
        roc_auc = auc(fpr, tpr)
        j_scores = tpr - fpr                    # Youden's J statistic
        best_idx = np.argmax(j_scores)           # Index of best threshold
        candidate_threshold = thresholds[best_idx]

        # Guard against degenerate cases (inf threshold or inverted AUC)
        if not np.isfinite(candidate_threshold) or roc_auc < 0.5:
            self.optimal_threshold = 0.5
            print(f"Warning: ROC AUC={roc_auc:.4f}, using default threshold 0.5")
        else:
            self.optimal_threshold = candidate_threshold

        # Save threshold alongside ensemble weights for reproducibility
        torch.save({
            'ensemble_state': self.ensemble_layer.state_dict(),
            'optimal_threshold': self.optimal_threshold,
        }, self.config["weights_save_path"])

        print(f"Optimal threshold: {self.optimal_threshold:.4f} (ROC AUC: {roc_auc:.4f})")
        self.plot_roc_curve(fpr, tpr, roc_auc, best_idx)

    def plot_training_loss(self, epoch_losses):
        """Saves a plot of validation loss vs. epochs to loss_vs_epoch.png"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-')
        plt.title('Validation Loss vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.grid(True)
        plt.savefig('loss_vs_epoch.png')
        print("Loss graph saved to 'loss_vs_epoch.png'")
        plt.close()

    def plot_roc_curve(self, fpr, tpr, roc_auc, best_threshold_idx):
        """Saves a plot of the ROC curve with the optimal threshold marked."""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Random classifier line
        plt.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx], marker='o', color='red', s=100,
                    label=f'Optimal Threshold ({self.optimal_threshold:.2f})')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right"); plt.grid(True)
        plt.savefig('roc_curve.png')
        print("ROC curve saved to 'roc_curve.png'")
        plt.close()

    def evaluate(self, test_scores, test_labels):
        """
        Evaluates the trained classifier on held-out test data.

        Loads the saved ensemble weights and threshold, then:
        1. Standardizes test scores using the saved scaler
        2. Runs through the ensemble layer
        3. Applies sigmoid to get probabilities
        4. Classifies using the learned threshold
        5. Reports accuracy, precision, recall, F1
        """
        try:
            checkpoint = torch.load(self.config["weights_save_path"], weights_only=False)
            self.ensemble_layer.load_state_dict(checkpoint['ensemble_state'])
            self.optimal_threshold = checkpoint['optimal_threshold']
            self.scaler = joblib.load('scaler.gz')
        except FileNotFoundError:
            print("Error: No saved weights or scaler found. Train first.")
            return

        self.ensemble_layer.eval()

        # Standardize test scores using the TRAINING scaler (no data leakage)
        scaled = self.scaler.transform(test_scores)
        scores_t = torch.FloatTensor(scaled).to(self.device)

        with torch.no_grad():
            logits = self.ensemble_layer(scores_t)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()

        # Classify: probability >= threshold → FAKE (1), else → REAL (0)
        pred_classes = (preds >= self.optimal_threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(test_labels, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, pred_classes, average='binary', zero_division=0
        )

        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS (Threshold = {self.optimal_threshold:.4f})")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"{'='*60}\n")

# =====================================================================================
# 6. MAIN EXECUTION BLOCK
# =====================================================================================
if __name__ == "__main__":

    # --- Configuration ---
    # Change these values to adjust the experiment.
    # 96% config: epochs=50, lr=0.01, patience=10, train_rows=500, test_rows=100
    # 94% config: epochs=200, lr=0.005, patience=20, train_rows=800, test_rows=200
    CONFIG = {
        "batch_size": 10,          # Articles per LLM inference batch
        "epochs": 200,             # Max training epochs for ensemble layer
        "learning_rate": 0.005,    # Adam optimizer learning rate
        "patience": 20,            # Early stopping: stop if no val improvement for N epochs
        "model_name": "Qwen/Qwen2.5-7B-Instruct",  # LLM used by all 4 agents
        "max_text_length": 1500,   # Max characters per article sent to LLM
        "weights_save_path": "best_ensemble_weights.pth",  # Where to save trained weights
        "train_rows": 800,         # Total rows to use (split equally between fake/real)
        "test_rows": 200,          # Held-out test set size
        "val_split": 0.2,          # Fraction of training data used for validation
    }

    classifier = AggregatedNewsClassifier(CONFIG)

    # ------------------------------------------------------------------
    # STEP 1: Load and prepare data from True.csv / Fake.csv
    # ------------------------------------------------------------------
    print("\nLoading datasets...")
    try:
        real_df = pd.read_csv('./Dataset/True.csv')
        fake_df = pd.read_csv('./Dataset/Fake.csv')
    except FileNotFoundError:
        print("Error: Dataset files not found. Ensure True.csv and Fake.csv are in ./Dataset/")
        exit()

    # Keep only the text column, assign binary labels (1 = FAKE, 0 = REAL)
    real_df = real_df[['text']].copy()
    real_df['label'] = 0
    fake_df = fake_df[['text']].copy()
    fake_df['label'] = 1

    # Balance the dataset: take equal samples from each class
    rows_per_class = (CONFIG["train_rows"] + CONFIG["test_rows"]) // 2
    real_sample = real_df.head(rows_per_class)
    fake_sample = fake_df.head(rows_per_class)

    # Shuffle and combine into a single DataFrame
    full_df = pd.concat([real_sample, fake_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train+val and test (stratified = equal fake/real ratio in each split)
    train_val_df, test_df = train_test_split(
        full_df, test_size=CONFIG["test_rows"], random_state=42, stratify=full_df['label']
    )
    # Split train into train and validation
    train_df, val_df = train_test_split(
        train_val_df, test_size=CONFIG["val_split"], random_state=42, stratify=train_val_df['label']
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ------------------------------------------------------------------
    # STEP 2: Collect agent scores ONCE (the expensive LLM step)
    # Model loads once, processes all 3 splits, then unloads.
    # ------------------------------------------------------------------
    train_dataset = NewsDataset(train_df)
    val_dataset = NewsDataset(val_df)
    test_dataset = NewsDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    (train_scores, train_labels), (val_scores, val_labels), (test_scores, test_labels) = \
        classifier.collect_all_scores(train_loader, val_loader, test_loader)

    # ------------------------------------------------------------------
    # STEP 3: Train ensemble weights (fast — no LLM needed, just 41 params)
    # ------------------------------------------------------------------
    classifier.train_weights(train_scores, train_labels, val_scores, val_labels)

    # ------------------------------------------------------------------
    # STEP 4: Evaluate on held-out test set (never seen during training)
    # ------------------------------------------------------------------
    classifier.evaluate(test_scores, test_labels)
