import os
import json
import re
from sympy import simplify
from sympy.parsing.latex import parse_latex
import sympy as sp

def sanitize_latex(expression: str) -> str:
    """
    Sanitize a LaTeX expression by fixing common issues and ensuring proper formatting.
    """
    # Replace common issues with LaTeX-safe expressions
    expression = expression.replace('%', r'\%')
    expression = expression.replace('$', r'\$')
    expression = expression.replace('∞', r'\infty')

    # Fix LaTeX interval format: Change only proper brackets to LaTeX form
    expression = re.sub(r'\(([\-\d\s\w]+),([\-\d\s\w]+)\)', r'\\left(\1, \2\\right)', expression)
    expression = re.sub(r'\[([\-\d\s\w]+),([\-\d\s\w]+)\]', r'\\left[\1, \2\\right]', expression)

    # Handle fraction formatting issues
    expression = re.sub(r'(\d+)\s*/\s*(\d+)', r'\\frac{\1}{\2}', expression)  # Example: change "1/2" to "\frac{1}{2}"

    return expression

def parse_matrix(latex_string):
    # Extract matrix entries from LaTeX pmatrix structure
    matrices = []
    matrix_pattern = re.compile(r'\\begin{pmatrix}(.*?)\\end{pmatrix}', re.DOTALL)
    matrix_matches = matrix_pattern.findall(latex_string)

    for match in matrix_matches:
        rows = match.split(r'\\\\')
        matrix = []
        for row in rows:
            entries = row.split('&')
            matrix.append([parse_latex(entry.strip()) for entry in entries])
        matrices.append(sp.Matrix(matrix))

    return matrices

def norm_parse(latex_string):
    # Regular expression to find the content inside \boxed{}
    matches = re.findall(r'\\boxed{((?:[^{}]+|{[^{}]*})*)}', latex_string)
    
    if matches:
        # Process all found boxed expressions
        answers = []
        for expression in matches:
            # Correct LaTeX escape sequences
            expression = expression.replace('\\\\', '\\')
            
            # Sanitize the LaTeX expression before parsing
            sanitized_expression = sanitize_latex(expression)
            
            try:
                # Check if it's a matrix and handle accordingly
                matrices = parse_matrix(sanitized_expression)
                if matrices:
                    # Assuming there's only one boxed answer in the solution for matrices
                    result_matrix = matrices[-1]  # Last matrix is typically the result in solutions
                    answers.append(str(result_matrix))
                else:
                    # Parse LaTeX to SymPy expression and simplify it
                    sympy_expr = parse_latex(sanitized_expression)
                    simplified_expr = simplify(sympy_expr)
                    
                    # Replace degree notation with actual degree symbol and add to answers
                    result = str(simplified_expr).replace('**circ', '°')
                    answers.append(result)
            except Exception as e:
                print(f"Error parsing LaTeX in file: {current_file}. Error: {e}")
                return None
        
        # Join all answers with commas (to handle cases with multiple answers)
        return ", ".join(answers)
    else:
        print(f"No boxed expression found in file: {current_file}")
        return None

def process_json_files(root_folder):
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                # Open and load the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract the solution and find the answer using norm_parse
                global current_file
                current_file = file_path
                if 'solution' in data:
                    solution = data['solution']
                    answer = norm_parse(solution)
                    
                    if answer:
                        # Add the 'answer' key to the JSON data
                        data['answer'] = answer

                        # Save the updated JSON back to the file
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                else:
                    print(f"No 'solution' found in file: {file_path}")
        
        print(f"Directory '{root}' processing completed successfully.")

# Specify the root directory containing your dataset
root_folder = '/content/drive/MyDrive/test_MATH'

# Process the JSON files in the root directory and its subdirectories
process_json_files(root_folder)
