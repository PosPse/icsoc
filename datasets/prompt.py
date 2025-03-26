class System:
    system_prompt = "You are an expert in business process management. You first thinks about the reasoning process in the mind and then provides the user with the answer."

class BusinessProcessDescriptionGeneration:
    instruction_prompt = "You will be given a title of business process description, and you need to generate a detailed business process description based on the title."
    input_prompt = "Title: {title}"
    output_prompt = "Business Process Description: {business_process_description}"
    
class ActorExtraction:
    instruction_prompt = "Extract all actors involved in the business process description."
    output_prompt = "Actors: {actors}"
    default_output_prompt = "There are no clear actors in this business process description."

class ActivityExtraction:
    instruction_prompt = "Extract the activities involved by all actors in the business process description."
    output_prompt = "Activities: {activities}"
    
if __name__ == "__main__":
    print(BusinessProcessDescriptionGeneration.input_prompt.format(title="Business Process Description"))