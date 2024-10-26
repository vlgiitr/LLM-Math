from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from huggingface_hub import login
import os
import json

from utils.metrics import combined_function
from utils.prompting_functions import get_sys_prompt, model_name, get_dir_adv_hint

login()

model_ids = [
  #  "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepseek-ai/deepseek-math-7b-instruct",
  #  "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-Math-7B-Instruct",
  #  "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mathstral-7B-v0.1",
    "google/gemma-2-2b-it"
    ]

dataset_dirs = [
    '../test_MATH/algebra',
    '../test_MATH/counting_and_probability',
    '../test_MATH/geometry/',
    '../test_MATH/intermediate_algebra/',
    '../test_MATH/number_theory/',
    '../test_MATH/prealgebra/',
    '../test_MATH/precalculus/'
    ]


eval_model_id = "Tianqiao/DeepSeek-7B-Math-Compare-Answer"
device = "cuda"  # the device to load the model onto

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
    input_text = input_text.replace("{{analysis}}", "") # default set analysis to blank, if exist, you can pass in the corresponding parameter.
    return input_text

model = AutoModelForCausalLM.from_pretrained(
    eval_model_id,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True
        ), 
    torch_dtype="auto", 
    device_map="auto",     
)
tokenizer = AutoTokenizer.from_pretrained(eval_model_id)
 
# Iterate through each JSON file in the dataset
for model_id in model_ids:
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
        hint = get_dir_adv_hint(dataset_dir)
        
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
                        Problem to be solved: {problem}
                        Hint: {hint}"""}
                        ]
                elif model_id == "meta-llama/Meta-Llama-3.1-8B":
                    messages = [
                        f"""{system_prompt}: 
                        Problem to be solved: {problem}
                        Hint: {hint}"""
                        ]
                elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
                    messages = [
                        f"""<s>[INST] Using this information : {system_prompt} 
                        Answer the Question : 
                        Problem to be solved: {problem}
                        Hint: {hint} [/INST]"""
                        ]
                else:
                    user_prompt = f"""Problem to be solved: {problem}
                    Hint: {hint}"""
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
        data_eval = correct_data_eval/total_data_eval
        model_results[f'{dataset_dir[9:-1]}'] = data_eval           
    model_eval = correct_model_eval/total_model_eval  
    model_results['final'] = model_eval
    modelname = model_name(model_id)       
    file_path = '/home/aniketa/vlg/copyright-project/shivank/transformers/LLM-Math/Evaluation/Dataset_score' + f'base_adv_hint_{modelname}.json'
    with open(file_path, 'w') as file:
        json.dump(model_results, file, indent=4)
    torch.cuda.empty_cache()

print("All models evaluated.")