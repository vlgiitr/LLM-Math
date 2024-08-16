import os
import json

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

def answer_keeper(root_folder):
    answers = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    answers.append({'q_id': f'{filename[:-5]}', 'answer': list(data.items())[3][1]})
    return answers

folders = [
    'MATH/test/algebra/',
    'MATH/test/counting_and_probability/',
    'MATH/test/geometry/',
    'MATH/test/intermediate_algebra/',
    'MATH/test/number_theory/',
    'MATH/test/prealgebra/',
    'MATH/test/precalculus/'
    ]

answer_sets = []

for folder in folders:
    answer_sets.append({'name': f'{folder[9:-1]}', 'answer_set': answer_keeper(folder)})

for answer_set in answer_sets:
    set_results = []
    for answer in answer_set:
        answer = normalize_answer(answer)
        set_results.append({
                    'q_id': answer.get('q_id', 'unknown'),
                    'answer': answer
                })
    answer_sets[answer_set['name']] = set_results
with open(os.path.join("MATH/cleaned_test"), 'w') as f:
    json.dump(answer_sets, f, indent=2)
