from transformers import pipeline
import torch
from huggingface_hub import login
import os
import json

login()

def problem_keeper(root_folder):
    problems = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    problems.append({'q_id': f'{filename[:-5]}', 'question': list(data.items())[0][1]}) 
    return problems

def normalize_answer(answer):
    from sympy import simplify
    """
    Normalizes a mathematical expression to a canonical form.
    
    Parameters:
    answer (str): The mathematical expression as a string.
    
    Returns:
    str: The normalized mathematical expression.
    """
    try:
        normalized = simplify(answer)
        return str(normalized)
    except Exception as e:
        print(f"Error normalizing answer: {e}")
        return answer


model_ids = [
    "mistralai/Mathstral-7B-v0.1",
    "meta-llama/Meta-Llama-3.1-8B",
    "Qwen/Qwen2-7B-Instruct", 
    "Qwen/Qwen2-Math-7B-Instruct", 
    "mistralai/Mistral-7B-Instruct-v0.3",
    "databricks/dolly-v2-12b",
    "stabilityai/stablelm-2-12b",
    "google/gemma-2-2b-it"
    ]

folders = [
    'MATH/test/algebra/',
    'MATH/test/counting_and_probability/',
    'MATH/test/geometry/',
    'MATH/test/intermediate_algebra/',
    'MATH/test/number_theory/',
    'MATH/test/prealgebra/',
    'MATH/test/precalculus/'
    ]

problem_sets = []
gen_answer_set = []

for folder in folders:
    problem_sets.append({'name': f'{folder[9:-1]}', 'problem_set': problem_keeper(folder)})

for model_id in model_ids:
    model_results = {}  
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "quantization_config": {"load_in_4bit": True}
        }
    )
    for problem_set in problem_sets:
        set_results = []
        for problem in problem_set:
            if model_id == "google/gemma-2-2b-it":
                messages = [
                    {"role": "user", "content": "Give one word singular numerical answer only, NO EXPLANATION, Do not use any LaTeX or MarkDown in the output, Solve the question: We flip a fair coin 10 times. What is the probability that we get heads in at least 6 of the 10 flips?"}
                    ]
            else:
                messages = [
                {"role": "system", "content": "Give one word singular numerical answer, NO EXPLANATION, Do not use any LaTeX or MarkDown in the output"},
                {"role": "user", "content": f"dkos{problem["question"]}"}
                ]
            outputs = pipe(
                messages,
                max_new_tokens=20,
                do_sample=False,
            trust_remote_code = True
            )
            assistant_response = outputs[0]["generated_text"][-1]["content"]
            assistant_response = normalize_answer(assistant_response)
            set_results.append({
                        'q_id': problem.get('q_id', 'unknown'),
                        'question': problem['question'],
                        'generated_answer': assistant_response.strip()
                    })
        model_results[problem_set['name']] = set_results

    #Save results for this model
    model_file_name = f"{model_id.replace('/', '_')}_results.json"
    with open(os.path.join("Generated_Responses/", model_file_name), 'w') as f:
        json.dump(model_results, f, indent=2)
    
    print(f"Results for {model_id} saved successfully.")


print("All processing complete. Results saved in the 'Results' directory.")