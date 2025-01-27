"""
Main entry point for the Self-Instruct pipeline.

This script orchestrates the full Self-Instruct process:
1. Generate instructions from seed tasks
2. Classify instructions as classification/generation tasks
3. Generate instances for each instruction
4. Filter and process the generated instances
5. Convert and save the final dataset

The pipeline creates a dataset of instruction-following examples that can be used
to fine-tune language models.
"""
import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
from typing import OrderedDict
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer

from gpt_api import make_requests as make_gpt_requests
from bootstrap_instructions import sample_machine_instructions, encode_prompt, post_process_response
from filter_instances import parse_instances_for_generation_task, parse_instances_for_classification_task, encode_instance
from utils import convert_alpaca_to_openai_format
# use en template
from templates.clf_task_template import template_en as template_clf
from templates.instance_gen_template import output_first_template_for_clf_en as output_first_template_for_clf
from templates.instance_gen_template import input_first_template_for_gen_en as input_first_template_for_gen


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the output is stored.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        default=None,
        help="The number of instructions after filtering."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()


def generate_instructions(args):
    """
    Generate new instructions by prompting GPT with seed tasks.
    
    Uses seed tasks and previously generated instructions to create new diverse 
    instruction-following tasks. Filters for novelty using ROUGE similarity.
    
    Args:
        args: Command line arguments
        
    Returns:
        list: Generated instruction data including similarity scores and metadata
    """
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.classification_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    os.makedirs(args.work_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.work_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.work_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(
            f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))
    results = []
    with open(os.path.join(args.work_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                prompt_instructions = sample_machine_instructions(
                    machine_instructions,
                    similarities=None,
                    n=2)
                # sample human instructions from the pool
                prompt_instructions += random.sample(
                    seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(
                    prompt_instructions, classification=args.classification_tasks_only)
                batch_inputs.append(prompt)
            results = make_gpt_requests(
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1
            )
            instructions = []
            all_metadata = []
            for result in results:
                new_instructions = post_process_response(
                    result["response"])
                instructions += new_instructions
                all_metadata += [result['response']['choices'][0].text] * len(new_instructions)
            for inst, metadata in zip(instructions, all_metadata):
                with Pool(4) as p:
                    rouge_scores = p.map(
                        partial(scorer.score, inst), seed_instructions + machine_instructions)
                rouge_scores = [
                    score["rougeL"].fmeasure for score in rouge_scores]
                if max(rouge_scores) > 0.7:
                    continue
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }
                machine_instructions.append(inst)
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,
                    "request_idx": request_idx
                }) + "\n")
                results.append({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,
                    "request_idx": request_idx
                })
                progress_bar.update(1)
            request_idx += 1
    return results


def classify_instructions(args):
    """
    Classify generated instructions as classification or generation tasks.
    
    Uses GPT to determine if each instruction represents a classification or
    generation task by prompting with templates.
    
    Args:
        args: Command line arguments
    """
    templates = {
        "template_clf": template_clf
    }

    with open(os.path.join(args.work_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()

    output_path = os.path.join(
        args.work_dir, f"is_clf_or_not.jsonl")
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(lines))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(lines), args.request_batch_size):
            batch = [json.loads(
                line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "is_classification"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                # prefix = compose_prompt_prefix(human_written_tasks, batch[0]["instruction"], 8, 2)
                prefix = template_clf
                prompts = [prefix + " " + d["instruction"].strip() +
                           "\n" + "Is it classification?" for d in batch]
                results = make_gpt_requests(
                    prompts=prompts,
                    max_tokens=3,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop_sequences=["\n", "Task"],
                    logprobs=1,
                    n=1,
                    best_of=1)
                for i in range(len(batch)):
                    data = batch[i]
                    if results[i]["response"] is not None:
                        data["is_classification"] = results[i]["response"]["choices"][0].text
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "is_classification"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))


def generate_instances(args):
    """
    Generate input-output instances for each instruction.
    
    Creates multiple example input-output pairs for each instruction by prompting
    GPT with templates specific to classification/generation tasks.
    
    Args:
        args: Command line arguments
    """
    with open(os.path.join(args.work_dir, "machine_generated_instructions.jsonl")) as fin:
        lines = fin.readlines()
        tasks = []
        for line in lines:
            data = json.loads(line)
            tasks.append(data)

    task_clf_types = {}
    with open(os.path.join(args.work_dir, f"is_clf_or_not.jsonl")) as fin:
        for line in fin:
            data = json.loads(line)
            task_clf_types[data["instruction"]] = data["is_classification"].strip() in [
                "Yes", "yes", "YES"]

    if args.classification_tasks_only:
        tasks = [task for task in tasks if task_clf_types[task["instruction"]]]

    if args.generation_tasks_only:
        tasks = [task for task in tasks if not task_clf_types[task["instruction"]]]

    output_path = os.path.join(
        args.work_dir, "machine_generated_instances.jsonl")
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "raw_instances", "instance_metadata",
                         "most_similar", "avg_similarity_score"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = []
                for task in batch:
                    if task_clf_types[task["instruction"]]:
                        prompt = output_first_template_for_clf + \
                            " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                    else:
                        prompt = input_first_template_for_gen + \
                            " " + task["instruction"].strip() + "\n"
                        prompts.append(prompt)
                results = make_gpt_requests(
                    prompts=prompts,
                    # because the clf template is longer, we need to decrease the max_tokens
                    max_tokens=300 if any(
                        task_clf_types[task["instruction"]] for task in batch) else 350,
                    temperature=0,
                    top_p=0,
                    frequency_penalty=0,
                    presence_penalty=1.5,
                    stop_sequences=[
                        f"Example {args.max_instances_to_generate + 1}", "Task:"],
                    logprobs=1,
                    n=1,
                    best_of=1)
                for i in range(len(batch)):
                    data = batch[i]
                    data["instance_metadata"] = results[i]['response']['choices'][0].text
                    if results[i]["response"] is not None:
                        data["raw_instances"] = results[i]["response"]["choices"][0].text
                    else:
                        data["raw_instances"] = ""
                    data = OrderedDict(
                        (k, data[k]) for k in
                        ["instruction", "raw_instances",
                         "most_similar", "avg_similarity_score", "instance_metadata"]
                    )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))


def filter_instances(args):
    """
    Filter and process the generated instruction instances.
    
    Processes raw GPT outputs into clean input-output pairs, removes duplicates,
    and optionally samples a subset of instructions. Converts data into training
    format.
    
    Args:
        args: Command line arguments
    """
    training_instances = []

    generated_tasks = []
    for instance_file in ["data/gpt3_generations/machine_generated_instances.jsonl"]:
        with open(instance_file) as fin:
            for line in fin:
                generated_tasks.append(json.loads(line))
    print(f"Loaded {len(generated_tasks)} raw generated tasks")

    task_clf_types = {}
    for file in ["data/gpt3_generations/is_clf_or_not.jsonl"]:
        with open(file) as fin:
            for line in fin:
                data = json.loads(line)
                task_clf_types[data["instruction"]] = data["is_classification"].strip() in [
                    "Yes", "yes", "YES"]

    for task in tqdm.tqdm(generated_tasks):
        # get instruction
        instruction = task["instruction"]
        task["is_classification"] = task_clf_types[instruction]

        # get the instances
        if task["is_classification"]:
            task_instances = parse_instances_for_classification_task(
                task["raw_instances"], instruction, task["instance_metadata"])
        else:
            task_instances = parse_instances_for_generation_task(
                task["raw_instances"], instruction, task["instance_metadata"])

        # we only allow max 5 instances per task
        task_instances = random.sample(
            task_instances, min(len(task_instances), 5))

        if not task_instances:
            continue

        training_instances += task_instances

    with open(os.path.join(args.work_dir, "all_generated_instances.jsonl"), "w") as fout:
        for instance in training_instances:
            fout.write(json.dumps({
                "instruction": instance[0],
                "input": instance[1],
                "output": instance[2],
            }) + "\n")
    print(f"Saved {len(training_instances)} instances")
    unique_instructions = set([it[0] for it in training_instances])
    print(f"Unique instructions: {len(unique_instructions)}")
    clf_instructions = [
        instruction for instruction in unique_instructions if task_clf_types[instruction]]
    print(f"Classification instructions: {len(clf_instructions)}")
    non_clf_instructions = [
        instruction for instruction in unique_instructions if not task_clf_types[instruction]]
    print(f"Non-classification instructions: {len(non_clf_instructions)}")

    if args.num_instructions is not None:
        print(f"Sampling {args.num_instructions} instructions")
        sampled_instructions = random.sample(
            unique_instructions, args.num_instructions)
        training_instances = [
            it for it in training_instances if it[0] in sampled_instructions]
        print(
            f"Only using {len(training_instances)} instances for these sampled instructions.")
        with open(os.path.join(args.work_dir, f"sampled_generated_instances_{args.num_instructions}.jsonl"), "w") as fout:
            for instance in training_instances:
                fout.write(json.dumps({
                    "instruction": instance[0],
                    "input": instance[1],
                    "output": instance[2],
                }) + "\n")

    # get the prompt and completion for training gpt3
    gpt_instances = []
    for instance in training_instances:
        # get input and do preprocessing
        inst_input = instance[1]
        # for some tasks, we check whether the input contains colon, and if so, we remove the part before the colon
        if random.random() < 0.5:
            colon_words = re.findall(r"(\w+):", inst_input)
            # if only one colon is found, we assume the instance only have one input and we remove the field name before the colon
            if len(set(colon_words)) == 1:
                inst_input = inst_input.split(":", 1)[1].strip()
            else:
                inst_input = inst_input.strip()
            # we also replace two consecutive new lines with one new line half of the time
            inst_input = inst_input.replace("\n\n", "\n")

        gpt_instances.append(encode_instance(
            instance[0], inst_input, instance[2]))

    # remove duplicates
    filtered_instances = []
    prompt_completion_set = set()
    for instance in gpt_instances:
        instance_pair = (instance["prompt"], instance["completion"])
        if instance_pair not in prompt_completion_set:
            prompt_completion_set.add(
                (instance["prompt"], instance["completion"]))
            filtered_instances.append(instance)
    gpt_instances = filtered_instances

    # shuffle
    random.shuffle(gpt_instances)
    with open(os.path.join(args.work_dir, f"gpt_generated_instances_{len(gpt_instances)}.jsonl"), "w") as fout:
        for instance in gpt_instances:
            fout.write(json.dumps({
                "prompt": instance["prompt"],
                "completion": instance["completion"],
            }) + "\n")


if __name__ == "__main__":
    args = parse_args()
    # step1: generate instructions
    generate_instructions(args)
    # step2: classify instructions
    classify_instructions(args)
    # # step3: generate instances
    generate_instances(args)
    # # step4 filter instances
    filter_instances(args)
    # step5: dump instances
    final_alpaca_data_path = os.path.join(args.work_dir, "all_generated_instances.jsonl")
    data_alpaca = []
    with open(final_alpaca_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_alpaca.append(json.loads(line))
    data_openai = convert_alpaca_to_openai_format(data_alpaca)
    with open(args.output_file, "a") as fout:
        json.dump(data_openai, fout, ensure_ascii=False, indent=4)
    
