from transformers import pipeline
import torch
from huggingface_hub import login
import os
import json

from utils.metrics import norm_parse

login()

model_ids = [
    "meta-llama/Meta-Llama-3.1-8B",
    "Qwen/Qwen2-7B-Instruct" 
    # "Qwen/Qwen2-Math-7B-Instruct", 
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "mistralai/Mathstral-7B-v0.1",
    # "databricks/dolly-v2-12b",
    # "stabilityai/stablelm-2-12b",
    # "google/gemma-2-2b-it"
    ]

dataset_dirs = [
    'trial/trial_folder_1/',
    'trial/trial_folder_2/'
    # 'test_MATH/algebra/',
    # 'test_MATH/counting_and_probability/',
    # 'test_MATH/geometry/',
    # 'test_MATH/intermediate_algebra/',
    # 'test_MATH/number_theory/',
    # 'test_MATH/prealgebra/',
    # 'test_MATH/precalculus/'
    ]

# Iterate through each JSON file in the dataset
for model_id in model_ids:
    model_results = {}
    correct_model_eval = 0 
    total_model_eval = 0 
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": {"load_in_4bit": True}
        }
    )
    for dataset_dir in dataset_dirs:
        correct_data_eval = 0 
        total_data_eval = 0
        for json_file in os.listdir(dataset_dir):
            if json_file.endswith('.json'):
                file_path = os.path.join(dataset_dir, json_file)

                # Load the JSON data
                with open(file_path, 'r') as file:
                    data = json.load(file)
                problem = data.get('problem', '')
                final_solution = data.get('final_solution', '')
                final_solution = norm_parse(final_solution)

                messages = [
                {"role": "system", "content": "You are an expert in Mathematics, you HAVE to give the final answer to the problem inside \\boxed{}, it is compulsory"},
                {"role": "user", "content": f"dkos{problem}"}
                ]
                outputs = pipe(
                    messages,
                    max_new_tokens=1024,
                    do_sample=False,
                trust_remote_code = True
                )

                # Extract the model's generated solution properly
                assistant_response = None
                try:
                    # Access the text attribute of the first candidate
                    assistant_response = outputs[0]["generated_text"][-1]["content"]
                    assistant_response = norm_parse(assistant_response)
                    
                except (IndexError, AttributeError) as e:
                    print(f"Error extracting solution for {json_file}: {e}")

                if assistant_response:
                    if assistant_response == final_solution:
                        correct_model_eval = +1
                        correct_data_eval = +1
                    total_model_eval =+1
                    total_data_eval =+1
                else:
                    print(f"No solution generated for {json_file}. Skipping.")
        data_eval = correct_data_eval/total_data_eval
        model_results[f'{dataset_dir[9:-2]}'] = data_eval           
    model_eval = correct_model_eval/total_model_eval  
    model_results['final'] = model_eval           
    file_path = 'Dataset_score/' + f'{model_id}.json'
    with open(file_path, 'w') as file:
        json.dump(model_results, file, indent=4)
print("All models evaluated.")