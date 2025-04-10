#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from config import Config
config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")

lora_path = "/home/user1/baitianrui/python/icsoc/sft/output/checkpoint-115"

# model = PeftModel.from_pretrained(model, lora_path)
#%%
messages = [
    {"role": "system", "content": "You are an expert in business process management. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
    {"role": "user", "content": "You will be given a title of business process description, and you need to generate a detailed business process description based on the title.\nTitle: \"PhD Student Expense Reimbursement and Funding Process\""},
    {"role": "assistant", "content": "Business Process Description: For PhD students, the process begins by collecting all the receipts. Once all the receipts have been collected, the process comes to an end. For PhD students, it is necessary to secure additional funds at the beginning. If the funds are 'not found,' then the student will have to use their own money. However, if the funds are 'found,' then the funding procedure can proceed. If the funds are 'not found,' then the student will have to use their own money. However, if the funds are 'found,' then the funding procedure can proceed and the process comes to an end. For the PhD office, the process begins by quickly reviewing the receipts. After reviewing the receipts, the process can continue. If the funds are deemed sufficient, the founds for travel can be removed. Alternatively, if the repayment amount exceeds the request but there are personal funds available, the founds for travel can be removed. However, if there is not enough money (found it!), then the process continues. If the funds are deemed sufficient, the founds for travel can be removed. Alternatively, if the repayment amount exceeds the request but there are personal funds available, the founds for travel can be removed. However, if there is not enough money (found it!), then the process continues. If the funds are deemed sufficient, the founds for travel can be removed. Alternatively, if the repayment amount exceeds the request but there are personal funds available, the founds for travel can be removed. However, if there is not enough money (found it!), then the process continues and comes to an end. For the Administration office, the process begins by checking all the information on the receipts. After checking the information, the amount of the repayment is calculated. Once the amount of the repayment has been calculated, the process can continue. The funds from the travel funder are then taken up, and the process comes to an end. For the Administration office, the process begins by making the repayment to the student's bank account. Once the repayment has been made, the process comes to an end."},
    {"role": "user", "content": "Extract all actors involved in the business process description. Extract the activities involved by all actors in the business process description."}
]
batch_size = 2
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt = [prompt] * batch_size
# print(prompt)
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
print(inputs)
# input_ids = inputs["input_ids"].to(model.device)
# attention_mask = inputs["attention_mask"].to(model.device)
# print(attention_mask)
# inputs = inputs.to(model.device)
# print(inputs.shape)
# with torch.no_grad():
#     output = model(input_ids=input_ids, attention_mask=attention_mask)
#     print(output.logits.shape)
#     print(output)
    # print(tokenizer.batch_decode(output, skip_special_tokens=False))
    # print(tokenizer.batch_decode(output, skip_special_tokens=True))
    # print(output)
    # print(tokenizer.decode(output[0], skip_special_tokens=False))

