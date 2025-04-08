class Config:
    model_path = "/home/user1/baitianrui/python/model/safetensors/Qwen2.5-0.5B-Instruct"
    ref_model_path = "/home/user1/baitianrui/python/model/safetensors/Qwen2.5-0.5B-Instruct"
    data_path = "/home/user1/baitianrui/python/icsoc/datasets/PAGED/instruction_tuning_dataset.json"
    output_dir = "./output"
    max_length = 4096
    generation_max_length = 2048
    epochs = 10
    batch_size = 1
    learning_rate = 1e-5
    step_size = 10
    gamma = 0.5
    do_sample = True
    top_p = 0.95
    temperature = 0.7
    beta = 0.1