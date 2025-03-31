class System:
    system_prompt = "You are an expert in business process management. You first thinks about the reasoning process in the mind and then provides the user with the answer."

class BusinessProcessDescriptionGeneration:
    instruction_prompt = "You will be given a title of business process description, and you need to generate a detailed business process description based on the title."
    input_prompt = "##Title:\n{title}"
    output_prompt = "##Business Process Description:\n{business_process_description}"
    
class ActorExtraction:
    instruction_prompt = "Extract all actors involved in the business process description."
    output_prompt = "##Actors:\n{actors}\n##The number of actors is {num_actors}."
    default_output_prompt = "##Actors:\nThere are no clear actors in this business process description. The number of actors is 0."

class ActivityExtraction:
    instruction_prompt = "Extract the activities involved by all actors in the business process description."
    output_prompt = "##Activities:\n{activities}"

class DataObjectExtraction:
    instruction_prompt = "Extract the data dependencies involved by all activities in the business process description."
    output_prompt = "##DataObjects:\n{data_objects}"
    default_output_prompt = "##DataObjects:\nThere are no data dependencies in the business process description."

class TextAnnotationExtraction:
    instruction_prompt = "Extract the text annotations involved by all activities in the business process description."
    output_prompt = "##TextAnnotations:\n{text_annotations}"
    default_output_prompt = "##TextAnnotations:\nThere are no text annotations in the business process description."

class InstructionTuning:
    system_prompt = "You are an expert in business process management. You first thinks about the reasoning process in the mind and then provides the user with the answer."
    instruction_prompt = "You will be given a business process description, and you need to extract the participants, activities, data objects, and text annotations involved in the business process description. At the same time, you need to analyze the logical relationships between them, including start, end, order, selection, exclusivity, inclusion, etc., and use BPMN like XML format to draw the final business process diagram."
    input_prompt = "##Business Process Description:\n{business_process_description}"
    
    @staticmethod
    def output_prompt(actors, num_actors, activities, data_objects, text_annotations, business_process_diagram):
        actors_prompt = ActorExtraction.output_prompt.format(actors=actors, num_actors=num_actors) if num_actors > 0 else ActorExtraction.default_output_prompt
        activities_prompt = ActivityExtraction.output_prompt.format(activities=activities)
        data_objects_prompt = DataObjectExtraction.output_prompt.format(data_objects=data_objects) if len(data_objects) > 0 else DataObjectExtraction.default_output_prompt
        text_annotations_prompt = TextAnnotationExtraction.output_prompt.format(text_annotations=text_annotations) if len(text_annotations) > 0 else TextAnnotationExtraction.default_output_prompt
        business_process_diagram_prompt = "##Business Process Diagram:\n" + business_process_diagram
        return f'''{actors_prompt}\n\n
{activities_prompt}\n\n
{data_objects_prompt}\n\n
{text_annotations_prompt}\n\n
{business_process_diagram_prompt}'''

if __name__ == "__main__":
    print(InstructionTuning.output_prompt(
        actors=["Actor1", "Actor2"],
        num_actors=2,
        activities=["Activity1", "Activity2"],
        data_objects=["DataObject1", "DataObject2"],
        text_annotations=["TextAnnotation1", "TextAnnotation2"],
        business_process_diagram="""<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.omg.org/spec/BPMN/20100524/MODEL BPMN20.xsd" id="Definitions_1" targetNamespace="http://bpmn.io/schema/bpmn" xmlns:bpmn20="http://www.omg.org/spec/BPMN/20100524/MODEL" xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI" xmlns:omgdc="http://www.omg.org/spec/DD/20100524/DC" xmlns:omgdi="http://www.omg.org/spec/DD/20100524/DI" id="Diagram_1" elementRe
        """
    ))
