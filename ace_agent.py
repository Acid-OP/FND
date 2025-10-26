import json
import os
import uuid
from sentence_transformers import SentenceTransformer, util

# Correctly import ONLY call_llm from your utility file
from utility import call_llm 

class ACEAgent:
    def __init__(self, agent_type, model, tokenizer, prompt_templates):
        self.agent_type = agent_type
        # Store the model and tokenizer directly
        self.model = model
        self.tokenizer = tokenizer
        self.playbook_path = f"playbooks/{agent_type}_playbook.json"
        self.prompts = prompt_templates
        # FIX: The original code had a typo here. This now correctly calls the internal method.
        self.playbook = self._load_playbook()
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # LOAD THE PLAYBOOK (Internal Method)
    def _load_playbook(self):
        if not os.path.exists(self.playbook_path):
            # Initialize with sections based on the paper's examples
            return {"strategies_and_hard_rules": [], "troubleshooting_and_pitfalls": []}
        with open(self.playbook_path, 'r') as f:
            return json.load(f)

    # SAVE THE PLAYBOOK
    def _save_playbook(self):
        with open(self.playbook_path, 'w') as f:
            json.dump(self.playbook, f, indent=2)

    # Corresponds to the Generator role
    def generate(self, input_data):
        prompt = self.prompts['generator'].format(
            playbook=json.dumps(self.playbook),
            input_data=json.dumps(input_data)
        )
        response_str = call_llm(prompt, self.model, self.tokenizer)
        return json.loads(response_str)

    # Corresponds to the Reflector role
    def reflect(self, trajectory, feedback):
        prompt = self.prompts['reflector'].format(
            trajectory=json.dumps(trajectory),
            feedback=json.dumps(feedback)
        )
        response_str = call_llm(prompt, self.model, self.tokenizer)
        return json.loads(response_str)

    # Corresponds to the Curator role
    def curate(self, insights):
        prompt = self.prompts['curator'].format(
            playbook=json.dumps(self.playbook),
            insights=json.dumps(insights)
        )
        response_str = call_llm(prompt, self.model, self.tokenizer)
        return json.loads(response_str)

    # Implements the grow-and-refine mechanism
    def update_playbook(self, delta_items):
        print(f"Updating playbook for {self.agent_type}...")
        for item in delta_items.get("operations", []):
            if item.get("type") == "ADD":
                section = item.get("section")
                new_content = item.get("content")
                
                # De-duplication step using semantic embeddings 
                is_redundant = False
                if section in self.playbook and self.playbook[section]:
                    existing_content = [entry['content'] for entry in self.playbook[section]]
                    new_embedding = self.similarity_model.encode(new_content)
                    existing_embeddings = self.similarity_model.encode(existing_content)
                    scores = util.cos_sim(new_embedding, existing_embeddings)[0]
                    if max(scores) > 0.95: # High similarity threshold
                        is_redundant = True
                        print(f"Skipping redundant content: '{new_content[:50]}...'")

                if not is_redundant and section in self.playbook:
                    self.playbook[section].append({
                        "id": f"{section[:3]}-{uuid.uuid4().hex[:6]}",
                        "content": new_content
                    })
        self._save_playbook()