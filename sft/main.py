from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from config import Config, LoraArguments
from datasets import load_dataset
from peft import LoraConfig
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator

config = Config()
lora_args = LoraArguments()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")

dataset = load_dataset("json", data_files=config.dataset_path, split="train")

max_length = 0
def process_func(example):
    global max_length
    input = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    encoding = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = input_ids.clone()
    max_length = max(max_length, len(input_ids[0]))
    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "labels": labels[0]
    }

dataset = dataset.map(process_func, remove_columns=dataset.column_names)
print(max_length)

splited = dataset.train_test_split(test_size=0.1)
train_dataset = splited["train"]
test_dataset = splited["test"]

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True))


lora_config = LoraConfig(
    r=lora_args.r,
    lora_alpha=lora_args.lora_alpha,
    lora_dropout=lora_args.lora_dropout,
    target_modules=lora_args.target_modules
)

model.add_adapter(lora_config)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
accelerator = Accelerator(log_with="tensorboard", project_dir="/home/user1/baitianrui/python/icsoc/sft/output")

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
model.train()

for epoch in range(3):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.log({"loss":loss})
        print(loss.item())
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
accelerator.end_training()

