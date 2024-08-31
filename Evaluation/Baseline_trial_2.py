from datasets import load_dataset
from transformers import pipeline
import torch
import os
import json

from utils.metrics import combined_function
from utils.prompting_functions import get_sys_prompt, model_name

model_ids = [
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # "deepseek-ai/deepseek-math-7b-instruct",
    # "Qwen/Qwen2-7B",
    # "Qwen/Qwen2-7B-Instruct",
    # "Qwen/Qwen2-Math-7B-Instruct",
    # "mistralai/Mistral-7B-v0.3",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Mathstral-7B-v0.1",
    # "google/gemma-2-2b-it"
    ]

dataset_splits = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

def modify_text(example, model_id, system_prompt):
    if model_id == "google/gemma-2-2b-it":
        example["problem"] = [
            {"role": "user", "content": f"""{system_prompt}:
            Problem: {example["problem"]}"""}
            ]
    elif model_id == "meta-llama/Meta-Llama-3.1-8B":
        example["problem"] = [
            f"""{system_prompt}: 
            Problem: {example["problem"]}"""
            ]
    elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
        example["problem"] = [
            f"""<s>[INST] Using this information : {system_prompt} 
            Answer the Question : 
            Problem: {example["problem"]} [/INST]"""
            ]
    else:
        user_prompt= f"""Problem: {example["problem"]}"""
        example["problem"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]
    user_prompt= f"""Problem: {example["problem"]}"""
    example["problem"] = [
        {"role": "system", "content": "ypu are pro"},
        {"role": "user", "content": user_prompt}
        ]
    return example

# Iterate through each JSON file in the dataset
for model_id in model_ids:
    #Loading model
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": {"load_in_4bit": True}
        }
    )
    dataset = load_dataset("yobro4619/test_MATH")
    system_prompt = get_sys_prompt(model_id)
    modified_dataset = dataset.map(modify_text, model_id, system_prompt)
    #iterating through datasets
    for dataset_split in dataset_splits:
        outputs = pipe(
            modified_dataset[f'{dataset_split}'],
            max_new_tokens=1024,
            do_sample=False,
        trust_remote_code = True
        )

        # Create a dictionary to store the data
        data = {
            "model_id": model_name(model_id),
            "split": dataset_split,
            "problem": dataset['problem'],
            "model_output": outputs,
            "solution": dataset['answer']
        }
        file_path = f"Evaluation/Outputs/full.json"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                main = json.load(file)
            main.extend([data])
            with open(file_path, 'w') as file:
                json.dump(main, file, indent=4)
        else:
            with open(file_path, 'w') as file:
                json.dump([data], file, indent=4)
print("Processing completed. JSON files saved.")