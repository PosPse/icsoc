from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from config import Config, LoraArguments
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from collections import defaultdict
import torch
import transformers
from packaging import version

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

# dataset = dataset.map(process_func)
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
    report_to="tensorboard",
)

model.add_adapter(lora_config)

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "eval" if self.control.should_evaluate else "train"
        (loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(torch.tensor(inputs["position_ids"].size(1))).sum().item()
                )
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            # Get predictions
            predictions = shift_logits.argmax(dim=-1)

            # Create mask for non-padding tokens (assuming ignore_index is -100)
            mask = shift_labels != -100

            # Calculate accuracy only on non-padding tokens
            correct_predictions = (predictions == shift_labels) & mask
            total_tokens = mask.sum()
            correct_tokens = correct_predictions.sum()

            # Gather the correct_tokens and total_tokens across all processes
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
            total_tokens = self.accelerator.gather_for_metrics(total_tokens)

            # Compute the mean token accuracy and log it
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs, start_time=None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()
    
# trainer = CustomTrainer(
#     model=model,
#     args=args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
# )

sft_config = SFTConfig(
    output_dir="/home/user1/baitianrui/python/icsoc/sft/output",
    max_seq_length=min(tokenizer.model_max_length, 4096), 
    per_device_train_batch_size=2, # by default 8
    per_device_eval_batch_size=2, # by default 8
    gradient_accumulation_steps=16,
    learning_rate=1e-4, # by default 5e-5
    weight_decay=0.1, # by default 0.0
    num_train_epochs=3, # by default 3
    logging_steps=1, # by default 500
    save_steps=5, # by default 500
    torch_empty_cache_steps=10,  # empty GPU cache every 10 steps
    eval_strategy='steps', # by default 'no'
    eval_steps=100, 
    bf16=True,
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=lora_config,
)

trainer.train()


