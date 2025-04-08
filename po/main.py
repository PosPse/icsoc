from config import Config
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from accelerate import Accelerator
import torch.nn.functional as F
from tqdm import tqdm

config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model_path)

def process_func(data):
    system_prompt = data["system"]
    instruction = data["instruction"]
    input = data["input"]
    output = data["output"]
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'{instruction}\n{input}'},
    ]
    chosen_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'{instruction}\n{input}'},
        {"role": "assistant", "content": output},
    ]
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    chosen_prompt = tokenizer.apply_chat_template(chosen_prompt, tokenize=False, add_generation_prompt=False)
    prompt_encoding = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=config.max_length, add_special_tokens=False)
    chosen_encoding = tokenizer(
        chosen_prompt, return_tensors="pt", padding=True, truncation=True, max_length=config.max_length, add_special_tokens=False)
    return {
        "prompt": prompt,
        "prompt_input_ids": prompt_encoding["input_ids"][0],
        "prompt_attention_mask": prompt_encoding["attention_mask"][0],
        "chosen_input_ids": chosen_encoding["input_ids"][0],
        "chosen_attention_mask": chosen_encoding["attention_mask"][0],
    }

dataset = load_dataset("json", data_files=config.data_path, split="train")
dataset = dataset.map(process_func, remove_columns=dataset.column_names)
print(dataset[0])
splited = dataset.train_test_split(test_size=0.1)
train_dataset = splited["train"]
test_dataset = splited["test"]

def collate_fn(batch):
    max_length = min(
        max(max([len(item["prompt_input_ids"]) for item in batch]),
        max([len(item["chosen_input_ids"]) for item in batch])
        ),
        config.max_length
    )
    
    batch_dict = {
        "prompt": [],
        "prompt_input_ids": [],
        "prompt_attention_mask": [],
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
    }

    for item in batch:
        prompt = item["prompt"]
        prompt_input_ids = item["prompt_input_ids"]
        prompt_attention_mask = item["prompt_attention_mask"]
        chosen_input_ids = item["chosen_input_ids"]
        chosen_attention_mask = item["chosen_attention_mask"]
        
        if len(prompt_input_ids) > max_length:
            prompt_input_ids = prompt_input_ids[:max_length]
            prompt_attention_mask = prompt_attention_mask[:max_length]
        else:
            padding_length = max_length - len(prompt_input_ids)
            prompt_input_ids = torch.tensor(prompt_input_ids + [tokenizer.pad_token_id] * padding_length)
            prompt_attention_mask = torch.tensor(prompt_attention_mask + [0] * padding_length)
        
        if len(chosen_input_ids) > max_length:
            chosen_input_ids = chosen_input_ids[:max_length]
            chosen_attention_mask = chosen_attention_mask[:max_length]
        else:
            padding_length = max_length - len(chosen_input_ids)
            chosen_input_ids = torch.tensor(chosen_input_ids + [tokenizer.pad_token_id] * padding_length)
            chosen_attention_mask = torch.tensor(chosen_attention_mask + [0] * padding_length)
        batch_dict["prompt"].append(prompt)
        batch_dict["prompt_input_ids"].append(prompt_input_ids)
        batch_dict["prompt_attention_mask"].append(prompt_attention_mask)
        batch_dict["chosen_input_ids"].append(chosen_input_ids)
        batch_dict["chosen_attention_mask"].append(chosen_attention_mask)

    return {
        "prompt": batch_dict["prompt"],
        "prompt_input_ids": torch.stack(batch_dict["prompt_input_ids"]),
        "prompt_attention_mask": torch.stack(batch_dict["prompt_attention_mask"]),
        "chosen_input_ids": torch.stack(batch_dict["chosen_input_ids"]),
        "chosen_attention_mask": torch.stack(batch_dict["chosen_attention_mask"]),
    }

def collate_fn_new(batch):
    max_length = min(
        max(max(len(item["chosen_input_ids"]) for item in batch),
        max(len(item["rejected_input_ids"]) for item in batch)
        ),
        config.max_length
    )
    batch_dict = {
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
    }

    for item in batch:
        chosen_input_ids = item["chosen_input_ids"]
        chosen_attention_mask = item["chosen_attention_mask"]
        rejected_input_ids = item["rejected_input_ids"]
        rejected_attention_mask = item["rejected_attention_mask"]
        
        if len(chosen_input_ids) > max_length:
            chosen_input_ids = chosen_input_ids[:max_length]
            chosen_attention_mask = chosen_attention_mask[:max_length]
        else:
            padding_length = max_length - len(chosen_input_ids)
            chosen_input_ids = torch.tensor(chosen_input_ids + [tokenizer.pad_token_id] * padding_length)
            chosen_attention_mask = torch.tensor(chosen_attention_mask + [0] * padding_length)

        if len(rejected_input_ids) > max_length:
            rejected_input_ids = rejected_input_ids[:max_length]
            rejected_attention_mask = rejected_attention_mask[:max_length]
        else:
            padding_length = max_length - len(rejected_input_ids)
            rejected_input_ids = torch.tensor(rejected_input_ids + [tokenizer.pad_token_id] * padding_length)
            rejected_attention_mask = torch.tensor(rejected_attention_mask + [0] * padding_length)

        batch_dict["chosen_input_ids"].append(chosen_input_ids)
        batch_dict["chosen_attention_mask"].append(chosen_attention_mask)
        batch_dict["rejected_input_ids"].append(rejected_input_ids)
        batch_dict["rejected_attention_mask"].append(rejected_attention_mask)

    return {
        "chosen_input_ids": torch.stack(batch_dict["chosen_input_ids"]),
        "chosen_attention_mask": torch.stack(batch_dict["chosen_attention_mask"]),
        "rejected_input_ids": torch.stack(batch_dict["rejected_input_ids"]),
        "rejected_attention_mask": torch.stack(batch_dict["rejected_attention_mask"]),
    }
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

class PolicyModel(nn.Module):
    def __init__(self, model_path, tokenizer, do_sample, top_p, temperature, generation_max_length, pad_token_id):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = tokenizer
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.generation_max_length = generation_max_length
        self.pad_token_id = pad_token_id
        

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask):
        config = GenerationConfig(do_sample=self.do_sample, 
                top_p=self.top_p,
                temperature=self.temperature, 
                max_new_tokens=self.generation_max_length,
                pad_token_id=self.pad_token_id,
            )
        rejected_input_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=config)
        completions = self.tokenizer.batch_decode(
                rejected_input_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        rejected_attention_mask = torch.ones_like(rejected_input_ids, dtype=torch.bool)
        rejected_attention_mask[rejected_input_ids == self.pad_token_id] = False
        return completions, rejected_input_ids, rejected_attention_mask

class ReferenceModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

policy_model = PolicyModel(config.model_path, tokenizer, config.do_sample, config.top_p, config.temperature, config.generation_max_length, tokenizer.eos_token_id)
reference_model = ReferenceModel(config.model_path)
policy_model.model.train()
reference_model.model.eval()
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
accelerator = Accelerator()
policy_model, reference_model, optimizer, scheduler = accelerator.prepare(policy_model, reference_model, optimizer, scheduler)
train_dataloader, test_dataloader = accelerator.prepare(train_dataloader, test_dataloader)

def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
    policy_logps = policy_chosen_logps - policy_rejected_logps
    reference_logps = reference_chosen_logps - reference_rejected_logps
    logits = policy_logps - reference_logps
    loss = -F.logsigmoid(config.beta * logits)

    chosen_rewards = (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (policy_rejected_logps - reference_rejected_logps).detach()  
    return loss.mean(), chosen_rewards.mean(), rejected_rewards.mean()

def compute_logprobs(logits, labels, mask=None):
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    logps = F.log_softmax(logits, dim=-1)
    select_logprobs = torch.gather(
        input=logps,
        dim=-1,
        index=labels.unsqueeze(1)
    ).squeeze(1)
    if mask is not None:
        mask = mask[:, 1:].clone()
        select_logprobs = select_logprobs * mask
        average_logprobs = select_logprobs.sum(-1) / mask.sum(-1)
        return average_logprobs
    else:
        return select_logprobs.mean(-1) 
    
def compute_batch_loss(batch, policy_model, reference_model):
    policy_chosen_logps = compute_logprobs(
        logits=policy_model(batch["chosen_input_ids"], batch["chosen_attention_mask"]).logits,
        labels=batch["chosen_input_ids"],
        mask=batch["chosen_attention_mask"]
    )
    reference_chosen_logps = compute_logprobs(
        logits=reference_model(batch["chosen_input_ids"], batch["chosen_attention_mask"]).logits,
        labels=batch["chosen_input_ids"],
        mask=batch["chosen_attention_mask"]
    )
    policy_rejected_logps = compute_logprobs(
        logits=policy_model(batch["rejected_input_ids"], batch["rejected_attention_mask"]).logits,
        labels=batch["rejected_input_ids"],
        mask=batch["rejected_attention_mask"]
    )
    reference_rejected_logps = compute_logprobs(
        logits=reference_model(batch["rejected_input_ids"], batch["rejected_attention_mask"]).logits,
        labels=batch["rejected_input_ids"],
        mask=batch["rejected_attention_mask"]
    )
    loss, chosen_rewards, rejected_rewards = dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
    return loss, chosen_rewards, rejected_rewards

def train_step(batch, policy_model, reference_model):
    prompt_input_ids = batch["prompt_input_ids"]
    prompt_attention_mask = batch["prompt_attention_mask"]
    chosen_input_ids = batch["chosen_input_ids"]
    chosen_attention_mask = batch["chosen_attention_mask"]

    _, rejected_input_ids, rejected_attention_mask = policy_model.generate(prompt_input_ids, prompt_attention_mask)
    
    new_batch = []
    for i in range(len(chosen_input_ids)):
        new_batch.append({
            "chosen_input_ids": chosen_input_ids[i].tolist(),
            "chosen_attention_mask": chosen_attention_mask[i].tolist(),
            "rejected_input_ids": rejected_input_ids[i].tolist(),
            "rejected_attention_mask": rejected_attention_mask[i].tolist()
        })
    batch = collate_fn_new(new_batch)
    loss, chosen_rewards, rejected_rewards = compute_batch_loss(batch, policy_model, reference_model)
    return loss, chosen_rewards, rejected_rewards

for epoch in range(config.epochs):
    print(f"开始第 {epoch+1}/{config.epochs} 轮训练")
    for batch in tqdm(train_dataloader):
        loss, chosen_rewards, rejected_rewards = train_step(batch, policy_model, reference_model)
        print(loss.item())
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()







