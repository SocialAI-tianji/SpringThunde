import time
import argparse
from datetime import datetime
import json
import tqdm
import os
import random
import openai
from openai import OpenAI

"""
Interface for making requests to OpenAI's GPT API.

This module provides functionality to:
- Make batch requests to GPT API
- Handle retries and errors
- Process and format responses
- Support different GPT models and parameters
- Cache responses to avoid duplicate requests
"""

openai_base_url = os.getenv("OPENAI_BASE_URL") if os.getenv(
    "OPENAI_BASE_URL") else "https://api.openai.com/v1"
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME") if os.getenv(
    "OPENAI_MODEL_NAME") is not None else "gpt-4o"
client = OpenAI(api_key=openai_api_key,
                base_url=openai_base_url,
                )


def make_requests(
        prompts, max_tokens, temperature, top_p,
        frequency_penalty, presence_penalty, stop_sequences, logprobs, n, best_of, retries=3, api_key=None):
    """
    Make requests to OpenAI's GPT API with specified parameters.
    
    Args:
        prompts (str or list): Single prompt or list of prompts to send to the API
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Controls randomness in generation (0-1)
        top_p (float): Controls diversity via nucleus sampling (0-1)
        frequency_penalty (float): Penalizes frequent tokens (-2 to 2)
        presence_penalty (float): Penalizes tokens already present (-2 to 2)
        stop_sequences (list): List of sequences where generation should stop
        logprobs (int): Number of most likely tokens to return probabilities for
        n (int): Number of completions to generate for each prompt
        best_of (int): Number of completions to generate server-side
        retries (int, optional): Number of retry attempts. Defaults to 3
        api_key (str, optional): OpenAI API key. Defaults to None
        
    Returns:
        list: List of dictionaries containing prompt, response and creation timestamp
    """
    response = None
    target_length = max_tokens
    retry_cnt = 0
    backoff_time = 30
    while retry_cnt <= retries:
        response = client.completions.create(model=model_name,
                                             prompt=prompts,
                                             max_tokens=target_length,
                                             temperature=temperature,
                                             top_p=top_p,
                                             frequency_penalty=frequency_penalty,
                                             presence_penalty=presence_penalty,
                                             stop=stop_sequences,
                                             logprobs=logprobs,
                                             n=n,
                                             best_of=best_of)
        break

    if isinstance(prompts, list):
        results = []
        for j, prompt in enumerate(prompts):
            data = {
                "prompt": prompt,
                "response": {"choices": response.choices[j * n: (j + 1) * n]} if response else None,
                "created_at": str(datetime.now()),
            }
            results.append(data)
        return results
    else:
        data = {
            "prompt": prompts,
            "response": response,
            "created_at": str(datetime.now()),
        }
        return [data]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The input file that contains the prompts to GPT3.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="The output file to save the responses from GPT3.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        help="The openai GPT3 engine to use.",
    )
    parser.add_argument(
        "--max_tokens",
        default=500,
        type=int,
        help="The max_tokens parameter of GPT3.",
    )
    parser.add_argument(
        "--temperature",
        default=0.7,
        type=float,
        help="The temprature of GPT3.",
    )
    parser.add_argument(
        "--top_p",
        default=0.5,
        type=float,
        help="The `top_p` parameter of GPT3.",
    )
    parser.add_argument(
        "--frequency_penalty",
        default=0,
        type=float,
        help="The `frequency_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--presence_penalty",
        default=0,
        type=float,
        help="The `presence_penalty` parameter of GPT3.",
    )
    parser.add_argument(
        "--stop_sequences",
        default=["\n\n"],
        nargs="+",
        help="The `stop_sequences` parameter of GPT3.",
    )
    parser.add_argument(
        "--logprobs",
        default=5,
        type=int,
        help="The `logprobs` parameter of GPT3"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="The `n` parameter of GPT3. The number of responses to generate."
    )
    parser.add_argument(
        "--best_of",
        type=int,
        help="The `best_of` parameter of GPT3. The beam size on the GPT3 server."
    )
    parser.add_argument(
        "--use_existing_responses",
        action="store_true",
        help="Whether to use existing responses from the output file if it exists."
    )
    parser.add_argument(
        "--request_batch_size",
        default=20,
        type=int,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()


if __name__ == "__main__":
    random.seed(123)
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # read existing file if it exists
    existing_responses = {}
    if os.path.exists(args.output_file) and args.use_existing_responses:
        with open(args.output_file, "r") as fin:
            for line in fin:
                data = json.loads(line)
                existing_responses[data["prompt"]] = data

    # do new prompts
    with open(args.input_file, "r") as fin:
        if args.input_file.endswith(".jsonl"):
            all_prompts = [json.loads(line)["prompt"] for line in fin]
        else:
            all_prompt = [line.strip().replace("\\n", "\n") for line in fin]

    with open(args.output_file, "w") as fout:
        for i in tqdm.tqdm(range(0, len(all_prompts), args.request_batch_size)):
            batch_prompts = all_prompts[i: i + args.request_batch_size]
            if all(p in existing_responses for p in batch_prompts):
                for p in batch_prompts:
                    fout.write(json.dumps(existing_responses[p]) + "\n")
            else:
                results = make_requests(
                    engine=args.engine,
                    prompts=batch_prompts,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    stop_sequences=args.stop_sequences,
                    logprobs=args.logprobs,
                    n=args.n,
                    best_of=args.best_of,
                )
                for data in results:
                    fout.write(json.dumps(data) + "\n")
