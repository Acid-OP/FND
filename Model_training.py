import torch
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import pandas as pd
from sklearn.model_selection import train_test_split


# LOAD BASE MODEL & TOKENIZER

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Move model to GPU explicitly
model = model.to(device)
print(f"Model loaded on device: {device}")

# PEFT CONFIG

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
peft_model = peft_model.to(device)  # Move PEFT model to GPU
peft_model.print_trainable_parameters()
print(f"PEFT model on device: {next(peft_model.parameters()).device}")


# DATASET

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = str(self.labels[idx])
        
        # Create prompt
        prompt = f"""### Instruction:
You are a text classification assistant. Your task is to analyze the tone of the provided news text and classify its stylistic trustworthiness.

- Classify as 'true' if the tone is trustworthy (neutral, formal, objective).
- Classify as 'false' if the tone is untrustworthy (sensationalist, emotionally charged, opinionated).

### Input:
{text}

### Response:
{label}"""
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # Return only input_ids and attention_mask
        # Labels will be created in the collator
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }



# COLLATOR

class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract input_ids and attention_mask
        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        
        # Find max length in this batch
        max_len = max(len(ids) for ids in input_ids_list)
        
        # Pad manually
        padded_input_ids = []
        padded_attention_mask = []
        
        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            padding_length = max_len - len(input_ids)
            
            # Pad input_ids with pad_token_id
            padded_input_ids.append(
                input_ids + [self.tokenizer.pad_token_id] * padding_length
            )
            
            # Pad attention_mask with 0
            padded_attention_mask.append(
                attention_mask + [0] * padding_length
            )
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long)
        }
        
        # Create labels (same as input_ids but with padding tokens masked)
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch



# MAIN EXECUTION

if __name__ == "__main__":
    print("Loading dataset...")

    fake_data = pd.read_csv("./Dataset/Fake.csv")
    fake_data["label"] = "false"

    real_data = pd.read_csv("./Dataset/True.csv")
    real_data["label"] = "true"

    fake_df = fake_data[["text", "label"]]
    real_df = real_data[["text", "label"]]
    full_df = pd.concat([fake_df, real_df]).reset_index(drop=True)

    train_df, eval_df = train_test_split(full_df, test_size=0.15, random_state=42)

    print(f"Training data size: {len(train_df)}")
    print(f"Validation data size: {len(eval_df)}")

    # Create datasets
    train_dataset = NewsDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=256
    )
    
    eval_dataset = NewsDataset(
        texts=eval_df["text"].tolist(),
        labels=eval_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=256
    )

    # Create collator
    collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    
    # TRAINER
    
    training_args = TrainingArguments(
        output_dir="./qwen-finetune",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        fp16=True,  # Enable mixed precision training on GPU
        dataloader_pin_memory=True,  # Enable pin_memory for faster GPU transfer
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator
    )

    print("\nTraining started....")
    trainer.train()

    
    # SAVE FINAL ADAPTER
    
    final_adapter_path = "./final_news_adapter"
    trainer.save_model(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)
    print(f"\nTraining completed. Adapter saved to {final_adapter_path}")