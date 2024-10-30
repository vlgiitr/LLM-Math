from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
import os
import json
import argparse
from pathlib import Path
from utils.metrics import combined_function
from utils.prompting_functions import get_sys_prompt, model_name, chain_of_thought_adverserial


def setup_args():
    parser = argparse.ArgumentParser(description='Evaluate math models on test datasets')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where output JSON files will be saved'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='../test_MATH/',
        help='Base directory containing the test datasets (default: ../test_MATH/)'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        required=True,
        help='Hugging Face API token'
    )
    return parser.parse_args()

def get_dataset_dirs(base_dir):
    """Get all subdirectories from the base dataset directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Dataset directory {base_dir} does not exist")
    
    return [
        str(d) for d in base_path.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]

def get_model_ids():
    return [
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "deepseek-ai/deepseek-math-7b-instruct",
        "Qwen/Qwen2-7B",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-Math-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mathstral-7B-v0.1",
        "google/gemma-2-2b-it"
    ]

chat_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>human
{}<|im_end|>
<|im_start|>gpt
"""

basic_prompt = """## 任务描述\n\n你是一个数学老师，学生提交了题目的解题步骤，你需要参考`题干`，`解析`和`答案`，判断`学生解题步骤`的结果是否正确。忽略`学生解题步骤`中的错误，只关注最后的答案。答案可能出现在`解析`中，也可能出现在`答案`中。\n\n## 输入内容\n\n题干:\n\n\n{{question}}\n\n\n解析:\n\n\n{{analysis}}\n\n\n\n答案:\n\n\n{{answer}}\n\n\n学生解题步骤:\n\n\n{{pred_step}}\n\n\n输出:"""
base_prompt = chat_prompt.format(basic_prompt)

def build_user_query(question, pred_answer, answer, base_prompt):
    input_text = base_prompt.replace("{{question}}", question)
    input_text = input_text.replace("{{pred_step}}", pred_answer)
    input_text = input_text.replace("{{answer}}", answer)
    input_text = input_text.replace("{{analysis}}", "")
    return input_text

def save_results(results, output_dir, model_name):
    """Save results to JSON file in the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'CoT_Adv_{model_name}.json')
    with open(output_path, 'w') as f:
        json.dump(results, file=f, indent=4)

def main():
    args = setup_args()
    
    login(token=args.hf_token)
    
    eval_model_id = "Tianqiao/DeepSeek-7B-Math-Compare-Answer"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        eval_model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True
            ), 
        torch_dtype="auto", 
        device_map="auto", 
    )
    tokenizer = AutoTokenizer.from_pretrained(eval_model_id)
    dataset_dirs = get_dataset_dirs(args.dataset_dir)
    # Iterate through each JSON file in the dataset
    for model_id in get_model_ids():
        print(f"\nEvaluating model: {model_id}")
        model_results = {}
        correct_model_eval = 0 
        total_model_eval = 0 

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
        for dataset_dir in dataset_dirs:
            correct_data_eval = 0 
            total_data_eval = 0

            # Loading System Prompts model-wise
            system_prompt = get_sys_prompt(model_id)
            CoT_prompt = chain_of_thought_adverserial(dataset_dir)

            for json_file in os.listdir(dataset_dir):
                if json_file.endswith('.json'):
                    file_path = os.path.join(dataset_dir, json_file)

                    # Load the JSON data
                    with open(file_path, 'r') as file:
                        data = json.load(file)
                    problem = data.get('problem', '')
                    final_solution = data.get('answer', '')
                    
                    # Feeding questions
                    if model_id == "google/gemma-2-2b-it":
                        messages = [
                            {"role": "user", "content": f"""{system_prompt}: 
                            {CoT_prompt}
                            Problem to be solved: {problem}"""}
                            ]
                    elif model_id == "meta-llama/Meta-Llama-3.1-8B":
                        messages = [
                            f"""{system_prompt}: 
                            {CoT_prompt}
                            Problem to be solved: {problem}"""
                            ]
                    elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
                        messages = [
                            f"""<s>[INST] Using this information : {system_prompt} 
                            Answer the Question : 
                            {CoT_prompt}
                            Problem to be solved: {problem}
                            [/INST]"""
                            ]
                    else:
                        user_prompt = f"""{CoT_prompt}
                        Problem to be solved: {problem}"""
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                            ]
                        
                    #Generating Outputs and extracting the answer
                    outputs = pipe(
                        messages,
                        max_new_tokens=1024,
                        do_sample=False
                    )       
                    # Extract the model's generated solution properly
                    assistant_response = None
                    try:
                        if model_id == "meta-llama/Meta-Llama-3.1-8B":
                            assistant_response = outputs[0][0]["generated_text"]
                        else:
                            # Access the text attribute of the first candidate
                            assistant_response = outputs[0]["generated_text"][-1]["content"]
                    except (IndexError, AttributeError) as e:
                        print(f"Error extracting solution for {json_file}: {e}")
                    #assistant_response = combined_function(f"""r'''{assistant_response}'''""")

                    #Confirming the answers
                    prompt = build_user_query(problem, assistant_response, final_solution, base_prompt)
                    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
                    generated_ids = model.generate(model_inputs.input_ids, temperature=0, max_new_tokens=16, eos_token_id=100005)
                    generated_ids = [
                        output_ids[len(input_ids) :]
                        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
                    if str(response) == "<correct><|im_end|>":
                        correct_model_eval += 1
                        correct_data_eval += 1
                    total_model_eval = total_model_eval+1
                    total_data_eval =total_data_eval+1
            data_eval = correct_data_eval / total_data_eval if total_data_eval > 0 else 0
            dataset_name = os.path.basename(dataset_dir)
            model_results[dataset_name] = data_eval

        model_eval = correct_model_eval / total_model_eval if total_model_eval > 0 else 0
        model_results['final'] = model_eval
        modelname = model_name(model_id)  
        save_results(model_results, args.output_dir, modelname)

        torch.cuda.empty_cache()
    
    print("All models evaluated successfully.")

if __name__ == "__main__":
    main()