from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from config import Config
config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")

lora_path = "/home/user1/baitianrui/python/icsoc/sft/output/checkpoint-141"

model = PeftModel.from_pretrained(model, lora_path)

messages = [
    {"role": "system", "content": "You are an expert in business process management. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
    {"role": "user", "content": "You will be given a title of business process description, and you need to generate a detailed business process description based on the title.\nTitle: \"PhD Student Expense Reimbursement and Funding Process\""}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(prompt, max_new_tokens=2048)
    #print(output)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

