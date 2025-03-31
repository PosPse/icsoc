import json
from dataclasses import dataclass
from dacite import from_dict
from tqdm import tqdm
from prompt import (
    System, 
    BusinessProcessDescriptionGeneration, 
    ActorExtraction,
    ActivityExtraction,
    DataObjectExtraction,
    TextAnnotationExtraction,
    InstructionTuning
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
    def __init__(self, raw_data_path, xml_data_path):
        self._raw_data = self._load_data(raw_data_path)
        self._xml_data = self._load_data(xml_data_path)

    def _load_data(self, data_path):
        return json.load(open(data_path, "r"))
    
    def _formatting_prompts_func(self, system="", instruction="", input="", output="") -> dict:
        return {
            "system": system,
            "instruction": instruction,
            "input": input,
            "output": output
        }
    
    def _str_formatting(self, str):
        return str.replace("\n", " ")
    
    def _extract_from_raw(self, data: Data) -> tuple[str, str, list[str], list[str]]:
        # 0. Extract file_index
        file_index = data.file_index
        # 1. Extract title
        title = data.title
        # 2. Extract paragraph
        paragraph = data.paragraph

        step_nodes = data.step_nodes
        data_nodes = data.data_nodes
        text_nodes = data.text_nodes

        # 3. Extract actors
        actors = [node.agent for node in step_nodes + data_nodes + text_nodes]
        actors = list(set(actors))
        actors = list(filter(lambda x: x != "", actors))
        actors_len = len(actors)

        # 4. Extract activities
        activity_nodes = filter(lambda x: x.type == "Activity", step_nodes)
        activities = [] if actors_len == 0 else {}
        if actors_len > 0:
            for node in activity_nodes:
                if node.agent not in activities:
                    activities[node.agent] = []
                activities[node.agent].append(node.NodeText)
            activities = {k: list(set(v)) for k, v in activities.items()}
        else:
            for node in activity_nodes:
                activities.append(node.NodeText)
            activities = list(set(activities))

        # 5. Extract DataObject
        data_objects = [] if actors_len == 0 else {}
        if actors_len > 0:
            for node in data_nodes:
                if node.agent not in data_objects:
                    data_objects[node.agent] = []
                data_objects[node.agent].append(node.NodeText)
            data_objects = {k: list(set(v)) for k, v in data_objects.items()}
        else:
            for node in data_nodes:
                data_objects.append(node.NodeText)
            data_objects = list(set(data_objects))

        # 6. Extract TextAnnotation
        text_annotations = [] if actors_len == 0 else {}
        if actors_len > 0:
            for node in text_nodes:
                if node.agent not in text_annotations:
                    text_annotations[node.agent] = []
                text_annotations[node.agent].append(node.NodeText)
            text_annotations = {k: list(set(v)) for k, v in text_annotations.items()}
        else:
            for node in text_nodes:
                text_annotations.append(node.NodeText)
            text_annotations = list(set(text_annotations))

        return file_index, title, paragraph, actors, activities, data_objects, text_annotations
    
    def _transfer_to_dialog_data(self, data: Data) -> list[dict]:
        example = []
        file_index, title, paragraph, actors, activities, data_objects, text_annotations = self._extract_from_raw(data)
        ## 1. Generate business process description
        system = System.system_prompt
        instruction = BusinessProcessDescriptionGeneration.instruction_prompt
        input = BusinessProcessDescriptionGeneration.input_prompt.format(title=title)
        output = BusinessProcessDescriptionGeneration.output_prompt.format(business_process_description=paragraph)
        example.append(self._formatting_prompts_func(system, instruction, input, output))

        ## 2. Extract actors
        instruction = ActorExtraction.instruction_prompt
        output = ActorExtraction.output_prompt.format(actors=actors, num_actors=len(actors)) if len(actors) > 0 else ActorExtraction.default_output_prompt
        example.append(self._formatting_prompts_func(instruction=instruction, output=output))

        ## 3. Extract activities
        instruction = ActivityExtraction.instruction_prompt
        output = ActivityExtraction.output_prompt.format(activities=activities)
        example.append(self._formatting_prompts_func(instruction=instruction, output=output))

        ## 4. Extract DataObjects
        is_empty_data_objects = len(data_objects) == 0
        instruction = DataObjectExtraction.instruction_prompt
        output = DataObjectExtraction.output_prompt.format(data_objects=data_objects) if not is_empty_data_objects else DataObjectExtraction.default_output_prompt
        example.append(self._formatting_prompts_func(instruction=instruction, output=output))

        ## 5. Extract TextAnnotations
        is_empty_text_annotations = len(text_annotations) == 0
        instruction = TextAnnotationExtraction.instruction_prompt
        output = TextAnnotationExtraction.output_prompt.format(text_annotations=text_annotations) if not is_empty_text_annotations else TextAnnotationExtraction.default_output_prompt
        example.append(self._formatting_prompts_func(instruction=instruction, output=output))
        return example
    
    def _transfer_to_instruction_data(self, data: list[dict]) -> dict:
        file_index, title, paragraph, actors, activities, data_objects, text_annotations = self._extract_from_raw(data)
        business_process_diagram = self._xml_data[str(file_index)]
        system = InstructionTuning.system_prompt
        instruction = InstructionTuning.instruction_prompt
        input = InstructionTuning.input_prompt.format(business_process_description=paragraph)
        output = InstructionTuning.output_prompt(actors, len(actors), activities, data_objects, text_annotations, business_process_diagram)
        return self._formatting_prompts_func(system, instruction, input, output)
    
    def _transfer_to_openai_sharegpt_data(self, data: list[dict]) -> dict:
        dialog_data = []
        for d in data:
            if d["system"] != "":
                dialog_data.append({"role": "system", "content": d["system"]})
            dialog_data.append({"role": "user", "content": f'{d["instruction"]}\n{d["input"]}' if d["input"] != "" else d["instruction"]})
            dialog_data.append({"role": "assistant", "content": d["output"]})
        return {"messages": dialog_data}

    def build_dialog_tuning_dataset(self, save_path):
        openai_sharegpt_dataset = []
        for data in tqdm(self._raw_data, desc="Dialog Dataset Processing: "):
            data = from_dict(data_class=Data, data=data)
            data = self._transfer_to_dialog_data(data)
            data = self._transfer_to_openai_sharegpt_data(data)
            openai_sharegpt_dataset.append(data)
        self.save_dataset(openai_sharegpt_dataset, save_path)

    def build_instruction_tuning_dataset(self, save_path):
        instruction_tuning_dataset = []
        for data in tqdm(self._raw_data, desc="Instruction Dataset Processing: "):
            data = from_dict(data_class=Data, data=data)
            data = self._transfer_to_instruction_data(data)
            instruction_tuning_dataset.append(data)
        self.save_dataset(instruction_tuning_dataset, save_path)
        
    def save_dataset(self, custom_dataset, save_path):
        with open(save_path, "w") as f:
            json.dump(custom_dataset, f, indent=4)

if __name__ == "__main__":
    raw_data_path = "./PAGED/raw_data.json"
    xml_data_path = "./PAGED/xml_data.json"
    data_process = DataProcess(raw_data_path, xml_data_path)
    # data_process.build_dialog_tuning_dataset(save_path="./PAGED/dialog_tuning_dataset.json")
    data_process.build_instruction_tuning_dataset(save_path="./PAGED/instruction_tuning_dataset.json")