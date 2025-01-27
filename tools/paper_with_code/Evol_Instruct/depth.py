base_instruction = "I want you act as a Prompt Rewriter.\r\n \
					Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., chatgpt and GPT4) a bit harder to handle.\r\n \
					But the rewritten prompt must be reasonable and must be understood and responded by humans.\r\n \
					Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
					You SHOULD complicate the given prompt using the following method: \r\n\
					{} \r\n\
					You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #The Given Prompt#. \r\n\
					'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#\r\n"


def createConstraintsPrompt(instruction):
	"""
	Creates a prompt that adds additional constraints or requirements to the given instruction.
	
	Args:
		instruction (str): The original instruction to be modified
		
	Returns:
		str: A formatted prompt requesting to add more constraints to the instruction
	"""
	prompt = base_instruction.format("Please add one more constraints/requirements into #The Given Prompt#'")
	prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt

def createDeepenPrompt(instruction):
	"""
	Creates a prompt that increases the depth and breadth of inquiries in the given instruction.
	
	Args:
		instruction (str): The original instruction to be modified
		
	Returns:
		str: A formatted prompt requesting to deepen the inquiry aspects of the instruction
	"""
	prompt = base_instruction.format("If #The Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased.")
	prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt

def createConcretizingPrompt(instruction):
	"""
	Creates a prompt that replaces general concepts with more specific ones in the given instruction.
	
	Args:
		instruction (str): The original instruction to be modified
		
	Returns:
		str: A formatted prompt requesting to make concepts more specific
	"""
	prompt = base_instruction.format("Please replace general concepts with more specific concepts.")
	prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt


def createReasoningPrompt(instruction):
	"""
	Creates a prompt that transforms simple thinking processes into explicit multiple-step reasoning.
	
	Args:
		instruction (str): The original instruction to be modified
		
	Returns:
		str: A formatted prompt requesting to add multiple-step reasoning requirements
	"""
	prompt = base_instruction.format("If #The Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.")
	prompt += "#The Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Rewritten Prompt#:\r\n"
	return prompt