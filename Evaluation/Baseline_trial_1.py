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

    #iterating through datasets
    for dataset_split in dataset_splits:
        dataset = load_dataset("yobro4619/test_MATH", split=f"{dataset_split}")
        # Loading System Prompts model-wise
        system_prompt = get_sys_prompt(model_id)
        for i, entry in enumerate(dataset):
            problem = entry['problem']
            solution = entry['answer']

            # Feeding questions
            if model_id == "google/gemma-2-2b-it":
                messages = [
                    {"role": "user", "content": f"""{system_prompt}:
                    Problem: {problem}"""}
                    ]
            elif model_id == "meta-llama/Meta-Llama-3.1-8B":
                messages = [
                    f"""{system_prompt}: 
                    Problem: {problem}"""
                    ]
            elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
                messages = [
                    f"""<s>[INST] Using this information : {system_prompt} 
                    Answer the Question : 
                    Problem: {problem} [/INST]"""
                    ]
            else:
                user_prompt: f"""Problem: {problem}"""
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    ]

            # Generate model output
            outputs = pipe(
                messages,
                max_new_tokens=1024,
                do_sample=False,
            trust_remote_code = True
            )

            # Extract the model's generated solution properly
            assistant_response = None
            if model_id == "meta-llama/Meta-Llama-3.1-8B":
                assistant_response = outputs[0][0]["generated_text"]
            else:
                # Access the text attribute of the first candidate
                assistant_response = outputs[0]["generated_text"][-1]["content"]
            output = combined_function(f"""r'''{assistant_response}'''""")

            # Create a dictionary to store the data
            data = {
                "model_id": model_name(model_id), 
                "problem": problem,
                "model_output": output,
                "solution": solution
            }
            file_path = f"Evaluation/Outputs/{dataset_split}/problem_{i}.json"
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