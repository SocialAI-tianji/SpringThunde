
base_instruction = "I want you act as a Prompt Creator.\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt.\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.\r\n\
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and must be understood and responded by humans.\r\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#\r\n"



def createBreadthPrompt(instruction):
	"""
	Creates a prompt that instructs an AI to generate a new prompt based on a given input prompt.
	The new prompt should be in the same domain but more rare/unique, while maintaining similar length and complexity.
	
	Args:
		instruction (str): The input prompt that will serve as inspiration for creating a new prompt
		
	Returns:
		str: A formatted prompt containing the base instructions and the given input prompt
	"""
	prompt = base_instruction
	prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Created Prompt#:\r\n"
	return prompt