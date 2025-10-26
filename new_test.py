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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib # For saving the scaler

# =====================================================================================
# 1. DATASET SETUP
# =====================================================================================
class NewsDataset(Dataset):
    """Standard PyTorch Dataset for loading news articles and labels."""
    def __init__(self, dataframe):
        self.data_frame = dataframe
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        text = row['news']
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
            # print(f"  [{agent_name}] Processing article {i+1}/{len(prompt_dicts)}...")
            
            input_length = inputs['input_ids'][i].shape[0]
            generated_tokens = outputs[i][input_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if not response_text.strip().endswith('}'):
                response_text += '}'
            
            score = self._extract_score(response_text)
            
            if score is None:
                # print(f"  Warning: Score extraction failed for {agent_name}. Defaulting to 0.5.")
                score = 0.5
            
            results.append({
                'score': score,
                'analysis': response_text,
            })
        return results

# =====================================================================================
# 3. SPECIALIST AGENT DEFINITIONS
# =====================================================================================
def create_llama3_prompt(system_prompt, user_prompt):
    """Creates a structured prompt for Llama 3 Instruct models."""
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

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
            full_prompt = create_llama3_prompt(system_prompt, user_prompt)
            prompts.append({'full_prompt': full_prompt, 'user_prompt_content': user_prompt})
        return prompts

# ... (Other agent classes are the same)
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
            full_prompt = create_llama3_prompt(system_prompt, user_prompt)
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
            full_prompt = create_llama3_prompt(system_prompt, user_prompt)
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
            full_prompt = create_llama3_prompt(system_prompt, user_prompt)
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
        self.scaler = StandardScaler() # <<< NEW: Scaler for standardizing inputs

    def get_agent_scores_for_epoch(self, dataloader):
        """Collects all agent scores for all batches in an epoch."""
        print(f"\n{'='*60}\nCOLLECTING AGENT SCORES\n{'='*60}\n")
        
        self.llm_manager._load_model()
        batch_results = []
        
        for batch_idx, (news, labels) in enumerate(dataloader):
            print(f"--- Batch {batch_idx + 1}/{len(dataloader)} ---")
            all_scores = {}
            for agent_key in self.agent_order:
                agent = self.agents[agent_key]
                prompt_dicts = agent._create_prompts(news, self.config["max_text_length"])
                results = self.llm_manager.run_inference(prompt_dicts, agent.agent_name)
                all_scores[agent_key] = np.array([res['score'] for res in results])
            
            scores_stacked = np.stack([all_scores[key] for key in self.agent_order], axis=1)
            batch_results.append({'scores': scores_stacked, 'labels': labels.clone().detach()})
        
        self.llm_manager._unload_model()
        return batch_results

    def train_weights(self, train_dataloader):
        """Trains the aggregation weights and determines the optimal threshold."""
        optimizer = torch.optim.Adam(self.ensemble_layer.parameters(), lr=self.config["learning_rate"])
        best_loss = float('inf')
        epoch_losses = []

        print(f"\n{'='*60}\nSTARTING WEIGHT TRAINING\n{'='*60}")
        
        # <<< NEW: First, get all scores to fit the scaler
        print("Preprocessing: Getting all training scores to fit the scaler...")
        initial_batch_results = self.get_agent_scores_for_epoch(train_dataloader)
        all_train_scores = np.vstack([res['scores'] for res in initial_batch_results])
        self.scaler.fit(all_train_scores)
        joblib.dump(self.scaler, 'scaler.gz') # Save the fitted scaler
        print("Scaler has been fitted on training data and saved.")

        for epoch in range(self.config["epochs"]):
            print(f"\n{'#'*60}\nEPOCH {epoch + 1}/{self.config['epochs']}\n{'#'*60}\n")
            
            epoch_loss = 0.0
            self.ensemble_layer.train()
            
            for batch_idx, batch_data in enumerate(initial_batch_results):
                # <<< NEW: Transform scores using the fitted scaler
                scaled_scores = self.scaler.transform(batch_data['scores'])
                scores_tensor = torch.FloatTensor(scaled_scores).to(self.device)
                labels_tensor = batch_data['labels'].float().view(-1, 1).to(self.device)
                
                aggregated_logits = self.ensemble_layer(scores_tensor)
                loss = self.criterion(aggregated_logits, labels_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(initial_batch_results)
            epoch_losses.append(avg_loss)
            print(f"\n--- EPOCH {epoch + 1} SUMMARY ---")
            print(f"Average Epoch Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.ensemble_layer.state_dict(), self.config["weights_save_path"])
                print(f"  ✓ New best weights saved! (Loss: {best_loss:.4f})")
        
        self.plot_training_loss(epoch_losses)
        self.find_optimal_threshold(train_dataloader)

    def plot_training_loss(self, epoch_losses):
        """Generates and saves a plot of training loss vs. epochs."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', linestyle='-')
        plt.title('Training Loss vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.xticks(range(1, len(epoch_losses) + 1))
        plt.savefig('loss_vs_epoch.png')
        print("\nTraining loss graph saved to 'loss_vs_epoch.png'")
        plt.close()
    
    def find_optimal_threshold(self, dataloader):
        """Finds the best classification threshold using the ROC curve."""
        print("\nFinding optimal threshold on training data using ROC curve...")
        self.ensemble_layer.load_state_dict(torch.load(self.config["weights_save_path"]))
        self.ensemble_layer.eval()

        all_labels = []
        all_preds = []

        batch_results = self.get_agent_scores_for_epoch(dataloader)
        all_scores = np.vstack([res['scores'] for res in batch_results])
        all_labels = np.concatenate([res['labels'].numpy() for res in batch_results])
        
        # <<< NEW: Use scaler to transform data
        scaled_scores = self.scaler.transform(all_scores)
        scores_tensor = torch.FloatTensor(scaled_scores).to(self.device)

        with torch.no_grad():
            aggregated_logits = self.ensemble_layer(scores_tensor)
            all_preds = torch.sigmoid(aggregated_logits).cpu().numpy().flatten()

        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)
        j_scores = tpr - fpr
        best_threshold_idx = np.argmax(j_scores)
        self.optimal_threshold = thresholds[best_threshold_idx]

        print(f"Optimal threshold found: {self.optimal_threshold:.4f} (Maximizes TPR-FPR)")
        self.plot_roc_curve(fpr, tpr, roc_auc, best_threshold_idx)

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
        print("ROC curve graph saved to 'roc_curve.png'")
        plt.close()

    def evaluate(self, dataloader):
        """Evaluates the classifier using the best saved weights and optimal threshold."""
        try:
            self.ensemble_layer.load_state_dict(torch.load(self.config["weights_save_path"]))
            # <<< NEW: Load the saved scaler
            self.scaler = joblib.load('scaler.gz')
        except FileNotFoundError:
            print("Warning: No saved model or scaler found. Cannot evaluate.")
            return
            
        self.ensemble_layer.eval()
        
        print(f"\n{'='*60}\nSTARTING EVALUATION ON TEST SET\n{'='*60}")
        
        batch_results = self.get_agent_scores_for_epoch(dataloader)
        all_scores = np.vstack([res['scores'] for res in batch_results])
        all_labels = np.concatenate([res['labels'].numpy() for res in batch_results])
        
        # <<< NEW: Transform test scores with the loaded scaler
        scaled_scores = self.scaler.transform(all_scores)
        scores_tensor = torch.FloatTensor(scaled_scores).to(self.device)

        with torch.no_grad():
            aggregated_logits = self.ensemble_layer(scores_tensor)
            all_preds = torch.sigmoid(aggregated_logits).cpu().numpy().flatten()
        
        pred_classes = (all_preds >= self.optimal_threshold).astype(int)

        accuracy = accuracy_score(all_labels, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, pred_classes, average='binary', zero_division=0)

        print(f"\n--- EVALUATION RESULTS (Threshold = {self.optimal_threshold:.4f}) ---")
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
        "learning_rate": 0.001, # Lowered learning rate for more stable learning
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "max_text_length": 1500,
        "weights_save_path": "best_ensemble_weights.pth",
        "train_rows": 1000, # Increased data for better scaling and training
        "test_rows": 100
    }

    classifier = AggregatedNewsClassifier(CONFIG)
    
    print("\nLoading and preparing datasets...")
    try:
        train_file = pd.read_csv('./Dataset/politifact_train.csv')
        test_file = pd.read_csv('./Dataset/politifact_test.csv')
    except FileNotFoundError:
        print("Error: Dataset files not found. Make sure 'True.csv' and 'Fake.csv' are in a './Dataset/' folder.")
        exit()

    train_df = train_file[['news','label']].copy()
    test_df = test_file[['news','label']].copy()
    
    final_train_df = train_df.head(CONFIG["train_rows"])
    final_test_df = test_df.head(CONFIG["test_rows"])

    print(final_test_df)
    print(final_train_df)

    # train_df = pd.concat([
    #     real_df.head(CONFIG["train_rows"] // 2),
    #     fake_df.head(CONFIG["train_rows"] // 2)
    # ])
    # test_df = pd.concat([
    #     real_df.iloc[CONFIG["train_rows"] // 2 : (CONFIG["train_rows"] + CONFIG["test_rows"]) // 2],
    #     fake_df.iloc[CONFIG["train_rows"] // 2 : (CONFIG["train_rows"] + CONFIG["test_rows"]) // 2]
    # ])
    
    train_dataset = NewsDataset(final_train_df)
    test_dataset = NewsDataset(final_test_df)
    
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total testing samples: {len(test_dataset)}")

    classifier.train_weights(train_dataloader)
    classifier.evaluate(test_dataloader)