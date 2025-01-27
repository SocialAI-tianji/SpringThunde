import json


def convert_alpaca_to_openai_format(data,
                                    system_prompt=""):
    """
    Convert data from Alpaca format to OpenAI conversation format.
    
    Args:
        data (list): List of dictionaries in Alpaca format containing 'instruction', 'input' and 'output' fields
        system_prompt (str, optional): System prompt to be added to each conversation. Defaults to empty string
        
    Returns:
        list: List of conversations in OpenAI format with system, user and assistant messages
    """
    conversations = []
    for item in data:

        if "input" in item and item["input"]:
            user_content = f"{item['instruction']}\n\n{item['input']}"
        else:
            user_content = item['instruction']
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": item["output"]
                }
            ]
        }
        conversations.append(conversation)

    return conversations


def convert_openai_to_alpaca_format(data):
    """
    Convert data from OpenAI conversation format to Alpaca format.
    
    Args:
        data (list): List of conversations in OpenAI format containing messages with roles and content
        
    Returns:
        list: List of dictionaries in Alpaca format with instruction, input and output fields
    """
    instructions = []
    for conversation in data:
        messages = conversation["messages"]

        instruction = ""
        input_text = ""
        output = ""

        for msg in messages:
            if msg["role"] == "user":
                content = msg["content"]
                parts = content.split("\n\n", 1)
                instruction = parts[0]
                input_text = parts[1] if len(parts) > 1 else ""
            elif msg["role"] == "assistant":
                output = msg["content"]

        instruction_item = {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
        instructions.append(instruction_item)

    return instructions


def dump_instructions(data,
                      output_file):
    """
    Save instruction data to a JSON file.
    
    Args:
        data (list): List of instructions to save
        output_file (str): Path to the output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    with open("data/alpaca_data_cleaned.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    conversations = convert_alpaca_to_openai_format(data)

    with open("data/alpaca_data_openai.json", 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=4)
