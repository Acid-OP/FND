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
    """Standard PyTorch Dataset for loading news articles and labels."""
    def __init__(self, dataframe):
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
    """Manages the LLM's lifecycle (load/unload) and classification logic."""
    def __init__(self, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Loads the LLM and tokenizer into memory with quantization."""
        if self.model is not None:
            return

        print(f"\nLoading {self.model_name} onto {self.device}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model_kwargs = {
            'low_cpu_mem_usage': True,
            'quantization_config': bnb_config,
            'device_map': 'auto',
            'max_memory': {0: "10GB", "cpu": "30GB"}
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        print("Model loaded successfully.")

    def _unload_model(self):
        """Unloads the model and clears memory."""
        if self.model is not None:
            print(f"\nUnloading {self.model_name}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("Model unloaded and memory cleared.")

    def _extract_score(self, response_text):
        """Robustly extracts score from a JSON object in the response text."""
        try:
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                score = float(data['score'])
                if 0 <= score <= 1:
                    return score
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  [Debug] JSON parsing failed. Error: {e}. Raw Response: '{response_text}'")

        return None

    def run_inference(self, prompt_dicts, agent_name, max_new_tokens=80, temperature=0.3):
        """Runs inference for a given batch of prompts, assuming the model is already loaded."""
        prompt_strings = [p['full_prompt'] for p in prompt_dicts]
        inputs = self.tokenizer(prompt_strings, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("}")
        ]

        results = []
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id
            )

        for i in range(len(outputs)):
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = outputs[i][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if not response_text.strip().endswith('}'):
                response_text += '}'

            score = self._extract_score(response_text)

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
def create_chat_prompt(system_prompt, user_prompt):
    """Creates a structured prompt for Qwen Instruct models."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

class NewsStyleClassifier:
    def __init__(self):
        self.agent_name = "Style Agent"

    def _create_prompts(self, news_list, max_text_len):
        system_prompt = "You are an expert news analyst. Your only output must be a single, valid JSON object with a 'score' field."
        user_template = """Rate the professionalism of this news article from 0 to 1.
0 = professional (objective, well-sourced)
1 = unprofessional (sensational, biased)

Article: {news}

JSON Response:"""
        prompts = []
        for news in news_list:
            user_prompt = user_template.format(news=news[:max_text_len])
            full_prompt = create_chat_prompt(system_prompt, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts

class NewsSentimentClassifier:
    def __init__(self):
        self.agent_name = "Sentiment Agent"

    def _create_prompts(self, news_list, max_text_len):
        system_prompt = "You are an expert news analyst. Your only output must be a single, valid JSON object with a 'score' field."
        user_template = """Rate the sentiment of this news article from 0 to 1.
0 = objective/neutral
1 = highly emotional/opinionated

Article: {news}

JSON Response:"""
        prompts = []
        for news in news_list:
            user_prompt = user_template.format(news=news[:max_text_len])
            full_prompt = create_chat_prompt(system_prompt, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts

class NewsVocabClassifier:
    def __init__(self):
        self.agent_name = "Vocab Agent"

    def _create_prompts(self, news_list, max_text_len):
        system_prompt = "You are an expert news analyst. Your only output must be a single, valid JSON object with a 'score' field."
        user_template = """Rate the vocabulary complexity of this news article from 0 to 1.
0 = simple/accessible
1 = complex/jargon-heavy

Article: {news}

JSON Response:"""
        prompts = []
        for news in news_list:
            user_prompt = user_template.format(news=news[:max_text_len])
            full_prompt = create_chat_prompt(system_prompt, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts

class NewsSemanticClassifier:
    def __init__(self):
        self.agent_name = "Semantic Agent"

    def _create_prompts(self, news_list, max_text_len):
        system_prompt = "You are an expert news analyst. Your only output must be a single, valid JSON object with a 'score' field."
        user_template = """Rate the semantic clarity of this news article from 0 to 1.
0 = direct/literal
1 = abstract/figurative

Article: {news}

JSON Response:"""
        prompts = []
        for news in news_list:
            user_prompt = user_template.format(news=news[:max_text_len])
            full_prompt = create_chat_prompt(system_prompt, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts

# =====================================================================================
# 4. ENSEMBLE LAYER
# =====================================================================================
class EnsembleLayer(nn.Module):
    """A linear layer to learn the optimal combination of agent scores."""
    def __init__(self, num_agents):
        super().__init__()
        self.linear = nn.Linear(num_agents, 1)

    def forward(self, x):
        return self.linear(x)

# =====================================================================================
# 5. ORCHESTRATOR CLASS
# =====================================================================================
class AggregatedNewsClassifier:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.llm_manager = BaseAgentClassifier(model_name=config["model_name"])

        self.agents = {
            "style": NewsStyleClassifier(), "sentiment": NewsSentimentClassifier(),
            "vocab": NewsVocabClassifier(), "semantic": NewsSemanticClassifier()
        }
        self.agent_order = ["style", "sentiment", "vocab", "semantic"]

        self.ensemble_layer = EnsembleLayer(num_agents=len(self.agents)).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimal_threshold = 0.5
        self.scaler = StandardScaler()

    def collect_agent_scores(self, dataloader):
        """Runs LLM inference ONCE and caches all agent scores for the entire dataloader."""
        print(f"\n{'='*60}\nCOLLECTING AGENT SCORES\n{'='*60}\n")

        self.llm_manager._load_model()

        all_scores_list = []
        all_labels_list = []

        for batch_idx, (news, labels) in enumerate(dataloader):
            print(f"--- Batch {batch_idx + 1}/{len(dataloader)} ---")
            batch_scores = {}
            for agent_key in self.agent_order:
                agent = self.agents[agent_key]
                prompt_dicts = agent._create_prompts(news, self.config["max_text_length"])
                results = self.llm_manager.run_inference(prompt_dicts, agent.agent_name)
                batch_scores[agent_key] = np.array([res['score'] for res in results])

            scores_stacked = np.stack([batch_scores[key] for key in self.agent_order], axis=1)
            all_scores_list.append(scores_stacked)
            all_labels_list.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

        self.llm_manager._unload_model()

        all_scores = np.vstack(all_scores_list)
        all_labels = np.concatenate(all_labels_list)

        print(f"\nCollected scores for {len(all_labels)} samples.")
        return all_scores, all_labels

    def train_weights(self, train_scores, train_labels, val_scores, val_labels):
        """Trains ensemble weights on pre-computed agent scores."""
        optimizer = torch.optim.Adam(self.ensemble_layer.parameters(), lr=self.config["learning_rate"])
        best_loss = float('inf')
        epoch_losses = []

        # Fit scaler on training scores only
        self.scaler.fit(train_scores)
        joblib.dump(self.scaler, 'scaler.gz')
        print("Scaler fitted on training data and saved to 'scaler.gz'.")

        scaled_train = self.scaler.transform(train_scores)
        train_scores_t = torch.FloatTensor(scaled_train).to(self.device)
        train_labels_t = torch.FloatTensor(train_labels).view(-1, 1).to(self.device)

        scaled_val = self.scaler.transform(val_scores)
        val_scores_t = torch.FloatTensor(scaled_val).to(self.device)
        val_labels_t = torch.FloatTensor(val_labels).view(-1, 1).to(self.device)

        print(f"\n{'='*60}\nSTARTING WEIGHT TRAINING\n{'='*60}")
        print(f"Epochs: {self.config['epochs']}, LR: {self.config['learning_rate']}")
        print(f"Train samples: {len(train_labels)}, Val samples: {len(val_labels)}\n")

        for epoch in range(self.config["epochs"]):
            # --- Train ---
            self.ensemble_layer.train()
            logits = self.ensemble_layer(train_scores_t)
            loss = self.criterion(logits, train_labels_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Validate ---
            self.ensemble_layer.eval()
            with torch.no_grad():
                val_logits = self.ensemble_layer(val_scores_t)
                val_loss = self.criterion(val_logits, val_labels_t).item()

            epoch_losses.append(val_loss)
            print(f"Epoch {epoch + 1:3d}/{self.config['epochs']}  |  Train Loss: {loss.item():.4f}  |  Val Loss: {val_loss:.4f}", end="")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.ensemble_layer.state_dict(), self.config["weights_save_path"])
                print(f"  << saved (best)")
            else:
                print()

        # Load best weights back
        self.ensemble_layer.load_state_dict(torch.load(self.config["weights_save_path"]))

        self.plot_training_loss(epoch_losses)
        self.find_optimal_threshold(val_scores, val_labels)

    def find_optimal_threshold(self, val_scores, val_labels):
        """Finds the best classification threshold using ROC curve on VALIDATION data."""
        print("\nFinding optimal threshold on validation data...")
        self.ensemble_layer.eval()

        scaled = self.scaler.transform(val_scores)
        scores_t = torch.FloatTensor(scaled).to(self.device)

        with torch.no_grad():
            logits = self.ensemble_layer(scores_t)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()

        fpr, tpr, thresholds = roc_curve(val_labels, preds)
        roc_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        self.optimal_threshold = thresholds[best_idx]

        # Save threshold alongside weights
        torch.save({
            'ensemble_state': self.ensemble_layer.state_dict(),
            'optimal_threshold': self.optimal_threshold,
        }, self.config["weights_save_path"])

        print(f"Optimal threshold: {self.optimal_threshold:.4f} (ROC AUC: {roc_auc:.4f})")
        self.plot_roc_curve(fpr, tpr, roc_auc, best_idx)

    def plot_training_loss(self, epoch_losses):
        """Generates and saves a plot of validation loss vs. epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-')
        plt.title('Validation Loss vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.grid(True)
        plt.xticks(range(1, len(epoch_losses) + 1))
        plt.savefig('loss_vs_epoch.png')
        print("Loss graph saved to 'loss_vs_epoch.png'")
        plt.close()

    def plot_roc_curve(self, fpr, tpr, roc_auc, best_threshold_idx):
        """Generates and saves a plot of the ROC curve."""
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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
        """Evaluates the classifier on pre-computed test scores."""
        try:
            checkpoint = torch.load(self.config["weights_save_path"])
            self.ensemble_layer.load_state_dict(checkpoint['ensemble_state'])
            self.optimal_threshold = checkpoint['optimal_threshold']
            self.scaler = joblib.load('scaler.gz')
        except FileNotFoundError:
            print("Error: No saved weights or scaler found. Train first.")
            return

        self.ensemble_layer.eval()

        scaled = self.scaler.transform(test_scores)
        scores_t = torch.FloatTensor(scaled).to(self.device)

        with torch.no_grad():
            logits = self.ensemble_layer(scores_t)
            preds = torch.sigmoid(logits).cpu().numpy().flatten()

        pred_classes = (preds >= self.optimal_threshold).astype(int)

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
    CONFIG = {
        "batch_size": 10,
        "epochs": 20,
        "learning_rate": 0.001,
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "max_text_length": 1500,
        "weights_save_path": "best_ensemble_weights.pth",
        "train_rows": 500,
        "test_rows": 100,
        "val_split": 0.2,
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
    full_df = pd.concat([real_sample, fake_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train+val and test
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
    # ------------------------------------------------------------------
    train_dataset = NewsDataset(train_df)
    val_dataset = NewsDataset(val_df)
    test_dataset = NewsDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    print("\n--- Collecting TRAIN scores ---")
    train_scores, train_labels = classifier.collect_agent_scores(train_loader)

    print("\n--- Collecting VALIDATION scores ---")
    val_scores, val_labels = classifier.collect_agent_scores(val_loader)

    print("\n--- Collecting TEST scores ---")
    test_scores, test_labels = classifier.collect_agent_scores(test_loader)

    # ------------------------------------------------------------------
    # STEP 3: Train ensemble weights (fast, no LLM needed)
    # ------------------------------------------------------------------
    classifier.train_weights(train_scores, train_labels, val_scores, val_labels)

    # ------------------------------------------------------------------
    # STEP 4: Evaluate on held-out test set
    # ------------------------------------------------------------------
    classifier.evaluate(test_scores, test_labels)
