import json
import random
import argparse
from multiprocessing import Pool
import multiprocessing

from breadth import createBreadthPrompt
from depth import (createConcretizingPrompt, createConstraintsPrompt,
                   createDeepenPrompt, createReasoningPrompt)
from openai_access import call_chatgpt
from tqdm import tqdm
from utils import convert_alpaca_to_openai_format, convert_openai_to_alpaca_format, dump_instructions


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default='alpaca_data_openai.json')
    parser.add_argument('--output_file', type=str,
                        default='alpaca_data_evol.json')
    parser.add_argument('--num_processes', type=int, default=4)
    return parser.parse_args()


def process_single_obj(cur_obj):
    instruction = cur_obj['instruction'].strip(
    ) + '\r\n' + cur_obj['input'].strip()

    evol_prompts = []
    evol_prompts.append(createConstraintsPrompt(instruction))
    evol_prompts.append(createDeepenPrompt(instruction))
    evol_prompts.append(createConcretizingPrompt(instruction))
    evol_prompts.append(createReasoningPrompt(instruction))
    evol_prompts.append(createBreadthPrompt(instruction))

    selected_evol_prompt = random.choice(evol_prompts)
    evol_instruction = call_chatgpt(selected_evol_prompt)
    answer = call_chatgpt(evol_instruction)

    return {"instruction": evol_instruction, "output": answer}


def main():
    args = args_parser()
    fr = open(args.input_file, 'r')

    all_objs = json.load(fr)
    all_objs = convert_openai_to_alpaca_format(all_objs)
    all_objs = all_objs[:4]

    with Pool(processes=args.num_processes) as pool:
        evol_objs = list(tqdm(
            pool.imap(process_single_obj, all_objs),
            total=len(all_objs)
        ))

    results_openai_format = convert_alpaca_to_openai_format(evol_objs)
    dump_instructions(results_openai_format, args.output_file)


if __name__ == "__main__":
    main()
