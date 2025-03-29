from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from config import Config, LoraArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

config = Config()
lora_args = LoraArguments()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")

dataset = load_dataset("json", data_files=config.dataset_path, split="train")

max_length = 0
def process_func(example):
    global max_length
    input = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    encoding = tokenizer(input, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = input_ids.clone()
    max_length = max(max_length, len(input_ids[0]))
    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "labels": labels[0]
    }

dataset = dataset.map(process_func)
print(max_length)

splited = dataset.train_test_split(test_size=0.1)
train_dataset = splited["train"]
test_dataset = splited["test"]

lora_config = LoraConfig(
    r=lora_args.r,
    lora_alpha=lora_args.lora_alpha,
    lora_dropout=lora_args.lora_dropout,
    target_modules=lora_args.target_modules
)

args = TrainingArguments(
    output_dir="/home/user1/baitianrui/python/icsoc/sft/output",
    per_device_train_batch_size=4, # by default 8
    per_device_eval_batch_size=4, # by default 8
    gradient_accumulation_steps=16,
    learning_rate=1e-4, # by default 5e-5
    weight_decay=0.1, # by default 0.0
    num_train_epochs=3, # by default 3
    logging_steps=1, # by default 500
    save_steps=100, # by default 500
    torch_empty_cache_steps=10,  # empty GPU cache every 10 steps
    bf16=True,
)

model.add_adapter(lora_config)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

# sft_config = SFTConfig(
#     output_dir="/home/user1/baitianrui/python/icsoc/sft/output",
#     max_seq_length=min(tokenizer.model_max_length, 4096), 
#     per_device_train_batch_size=4, # by default 8
#     per_device_eval_batch_size=4, # by default 8
#     gradient_accumulation_steps=16,
#     learning_rate=1e-4, # by default 5e-5
#     weight_decay=0.1, # by default 0.0
#     num_train_epochs=3, # by default 3
#     logging_steps=1, # by default 500
#     save_steps=100, # by default 500
#     torch_empty_cache_steps=10,  # empty GPU cache every 10 steps
#     eval_strategy='steps', # by default 'no'
#     eval_steps=100, 
#     bf16=True,
# )

# trainer = SFTTrainer(
#     model=model,
#     args=sft_config,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     peft_config=lora_config,
# )

# trainer.train()


