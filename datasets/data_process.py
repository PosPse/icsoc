import json
from dataclasses import dataclass
from dacite import from_dict
from tqdm import tqdm
from prompt import (
    System, 
    BusinessProcessDescriptionGeneration, 
    ActorExtraction,
    ActivityExtraction,
    )

@dataclass
class Node:
    resourceId: str
    NodeText: str
    agent: str
    type: str

@dataclass
class SequenceFlow:
    src: str
    tgt: str
    condition: str

@dataclass
class Flow:
    src: str
    tgt: str

@dataclass
class Data:
    file_index: int
    paragraph: str
    step_nodes: list[Node]
    data_nodes: list[Node]
    text_nodes: list[Node]
    SequenceFlow: list[SequenceFlow]
    MessageFlow: list[Flow]
    Association: list[Flow]
    or_node_num: int
    title: str

class DataProcess:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = self.load_data()
        

    def load_data(self):
        raw_data = json.load(open(self.data_path, "r"))
        return raw_data
    
    def formatting_prompts_func(self, system="", instruction="", input="", output="") -> dict:
        return {
            "system": system,
            "instruction": instruction,
            "input": input,
            "output": output
        }
    
    def str_formatting(self, str):
        return str.replace("\n", " ")
    
    def extract_metadata(self, data: Data) -> tuple[str, str, list[str], list[str]]:
        # 1. Extract title
        title = data.title
        # 2. Extract paragraph
        paragraph = data.paragraph

        step_nodes = data.step_nodes
        data_nodes = data.data_nodes
        text_nodes = data.text_nodes

        # 3. Extract actors
        actors = []
        for node in step_nodes:
            actors.append(node.agent)
        for node in data_nodes:
            actors.append(node.agent)
        for node in text_nodes:
            actors.append(node.agent)
        actors = list(set(actors))
        actors = [i for i in actors if i != ""]

        # 4. Extract activities
        activities = []
        
        return title, paragraph, actors, activities
    
    def transfer_to_dialog_data(self, data: Data) -> list[dict]:
        example = []
        title, paragraph, actors, activities = self.extract_metadata(data)
        ## 1. Generate business process description
        system = System.system_prompt
        instruction = BusinessProcessDescriptionGeneration.instruction_prompt
        input = BusinessProcessDescriptionGeneration.input_prompt.format(title=title)
        output = BusinessProcessDescriptionGeneration.output_prompt.format(business_process_description=paragraph)
        example.append(self.formatting_prompts_func(system, instruction, input, output))

        ## 2. Extract actors
        instruction = ActorExtraction.instruction_prompt
        output = ActorExtraction.output_prompt.format(actors=actors) if len(actors) > 0 else ActorExtraction.default_output_prompt
        example.append(self.formatting_prompts_func(instruction=instruction, output=output))

        ## 3. Extract activities
        # instruction = ActivityExtraction.instruction_prompt
        # output = ActivityExtraction.output_prompt.format(activities=activities) if len(activities) > 0 else ActivityExtraction.default_output_prompt
        # example.append(self.formatting_prompts_func(instruction=instruction, output=output))

        ## 
        return example
    
    def transfer_to_openai_sharegpt_data(self, data: list[dict]) -> dict:
        dialog_data = []
        for d in data:
            if d["system"] != "":
                dialog_data.append({"role": "system", "content": d["system"]})
            dialog_data.append({"role": "user", "content": f'{d["instruction"]}\n{d["input"]}' if d["input"] != "" else d["instruction"]})
            dialog_data.append({"role": "assistant", "content": d["output"]})
        return {"messages": dialog_data}

    def build_dialog_tuning_dataset(self, save_path):
        openai_sharegpt_dataset = []
        for data in tqdm(self.raw_data, desc="Dialog Dataset Processing: "):
            data = from_dict(data_class=Data, data=data)
            data = self.transfer_to_dialog_data(data)
            data = self.transfer_to_openai_sharegpt_data(data)
            openai_sharegpt_dataset.append(data)
        self.save_dataset(openai_sharegpt_dataset, save_path)
        
    def save_dataset(self, custom_dataset, save_path):
        with open(save_path, "w") as f:
            json.dump(custom_dataset, f, indent=4)

if __name__ == "__main__":
    raw_data_path = "./PAGED/raw_data.json"
    data_process = DataProcess(raw_data_path)
    data_process.build_dialog_tuning_dataset(save_path="./PAGED/dialog_tuning_dataset.json")