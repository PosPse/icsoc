import torch

class Config:
    model_name = "/home/user1/baitianrui/python/model/safetensors/Qwen2.5-0.5B-Instruct"
    dataset_path = "/home/user1/baitianrui/python/icsoc/datasets/PAGED/dialog_tuning_dataset.json"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
class LoraArguments:
    r = 8
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
