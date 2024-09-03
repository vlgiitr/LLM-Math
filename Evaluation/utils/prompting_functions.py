import os
import json
import random

def model_name(model_id):
    model = ""
    if model_id == "meta-llama/Meta-Llama-3.1-8B":
        model= 'Meta-Llama-3.1-8B'
    elif model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        model= "Meta-Llama-3.1-8B-Instruct"
    elif model_id == "deepseek-ai/deepseek-math-7b-instruct":
        model= "deepseek-math-7b-instruct"
    elif model_id == "Qwen/Qwen2-7B":
        model= "Qwen2-7B"
    elif model_id == "Qwen/Qwen2-7B-Instruct":
        model= "Qwen2-7B-Instruct"
    elif model_id == "Qwen/Qwen2-Math-7B-Instruct":
        model= "Qwen2-Math-7B-Instruct"
    elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
        model= "Mistral-7B-Instruct-v0.3"
    elif model_id == "mistralai/Mathstral-7B-v0.1":
        model= "Mathstral-7B-v0.1"
    elif model_id == "google/gemma-2-2b-it":
        model= "gemma-2-2b-it"
    return (model)

def get_sys_prompt(model_id):
    if model_id == "meta-llama/Meta-Llama-3.1-8B":
        system_prompt = ''''Given the following math problem, provide a complete and detailed solution.
        Conclude by outputting the final answer after converting into plain text in next line in the format of dictionary key-value pair : {"answer": "final answer"}.
        The Problem is:'''
    elif model_id == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        system_prompt = '''Given the following math problem, provide a complete final solution but optimum.
        DON'T USE Boxed AND Latex, Conclude by outputting the final answer in last line in the following dictionary format:
        {'answer': final answer in simple number} '''
    elif model_id == "deepseek-ai/deepseek-math-7b-instruct":
        system_prompt = '''Given the following math problem, provide a complete and detailed solution.
        Conclude by outputting the final answer after converting into boxed format in LaTeX in next line in the format of dictionary key-value pair :
        "$\boxed{final answer}" $'''
    elif model_id == "Qwen/Qwen2-7B":
        system_prompt = '''give the answer of  as a text file'''
    elif model_id == "Qwen/Qwen2-7B-Instruct":
        system_prompt = '''Given the following math problem, provide a complete and detailed solution.
        Do not use LaTeX or markdown in your solution, it is compulsory to convert them into plain text.
        Conclude by outputting the final answer after converting into plain text from LaTeX in next line in the format of dictionary key-value pair :
            {"answer": final answer in number} strictly in this format'''
    elif model_id == "Qwen/Qwen2-Math-7B-Instruct":
        system_prompt = '''Given the following math problem, provide a complete final solution but optimum.
        Conclude by outputting the final answer after converting into boxed format in LaTeX in last line in the following format :
            "$\boxed{final answer}$" '''
    elif model_id == "mistralai/Mistral-7B-Instruct-v0.3":
        system_prompt = '''Given the following math problem, provide a complete final solution but optimum.
        DON'T USE Boxed AND Latex, Conclude by outputting the final answer in last line in the following dictionary format:
        {'answer': final answer} strictly in this format'''
    elif model_id == "mistralai/Mathstral-7B-v0.1":
        system_prompt = '''Given the following math problem, provide a complete final solution but optimum.
        Conclude by outputting the final answer after converting into boxed format in LaTeX in next line in the following format :
            "$\boxed{final answer}$"'''
    elif model_id == "google/gemma-2-2b-it":
        system_prompt = '''Give complete and detailed solution with all calculations for the following math problem. Conclude by outputting the final answer as a numerical value and in dictionary key-value pair format on a new line like: {'answer':'final numerical answer'}.
        Solve the question '''
    return (system_prompt)

def get_dataset_directory(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        dataset_dir = "train_MATH/algebra/"
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        dataset_dir = "train_MATH/counting_and_probability/"
    elif dataset_dir == 'test_MATH/geometry/':
        dataset_dir = "train_MATH/geometry/"
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        dataset_dir = "train_MATH/intermediate_algebra/"
    elif dataset_dir == 'test_MATH/number_theory/':
        dataset_dir = "train_MATH/number_theory/"
    elif dataset_dir == 'test_MATH/prealgebra/':
        dataset_dir = "train_MATH/prealgebra/"
    elif dataset_dir == 'test_MATH/precalculus/':
        dataset_dir = "train_MATH/precalculus/"
    return (dataset_dir)

def get_one_shot_prompt(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        example_part = "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?"
        answer = '5'
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        example_part = "What is the coefficient of $x^8$ in the expansion of $(x-1)^9$?"
        answer = '-9'
    elif dataset_dir == 'test_MATH/geometry/':
        example_part = "A cube with an edge length of 4 units has the same volume as a square-based pyramid with base edge lengths of 8 units and a height of $h$ units. What is the value of $h$?"
        answer = '3'
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        example_part = "Let $a$ and $b$ be nonzero real numbers such that\n\\[(2 - 7i)(a + bi)\\]is pure imaginary.  Find $\\frac{a}{b}.$"
        answer = '7/2'
    elif dataset_dir == 'test_MATH/number_theory/':
        example_part = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
        answer = '2'
    elif dataset_dir == 'test_MATH/prealgebra/':
        example_part = "Find $r$ if $3(r-7) = 4(2-2r) + 4$."
        answer = '3'
    elif dataset_dir == 'test_MATH/precalculus/':
        example_part = "Let $S$ be a region in the plane with area 10.  When we apply the matrix\n\\[\\begin{pmatrix} 2 & 1 \\\\ 7 & -3 \\end{pmatrix}\\]to $S,$ we obtain the region $S'.$  Find the area of $S'."
        answer = '130'
    one_shot_prompt = f'''Example Problem: {example_part}
        Answer for the example: {answer}
'''
    return (one_shot_prompt)

def get_one_shot_hint_prompt(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        example_part = "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?"
        hint = "To solve this problem, first find the values of *y* that make the denominator of the expression equal to zero.  Then, use the relationship between the coefficients of a quadratic equation and the sum of its roots to determine the sum of these values of *y*. \n"
        answer = '5'
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        example_part = "What is the coefficient of $x^8$ in the expansion of $(x-1)^9$?"
        hint = "To find the coefficient of $x^8$, use the Binomial Theorem to expand the expression and identify the term containing $x^8$. Calculate the binomial coefficient and the coefficient of the remaining factor to determine the overall coefficient of $x^8$. \n"
        answer = '-9'
    elif dataset_dir == 'test_MATH/geometry/':
        example_part = "A cube with an edge length of 4 units has the same volume as a square-based pyramid with base edge lengths of 8 units and a height of $h$ units. What is the value of $h$?"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
        answer = '3'
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        example_part = "Let $a$ and $b$ be nonzero real numbers such that\n\\[(2 - 7i)(a + bi)\\]is pure imaginary.  Find $\\frac{a}{b}.$"
        hint = "To find the ratio a/b, we need to expand the given expression, equate the real part to zero, and then solve for the ratio a/b. \n"
        answer = '7/2'
    elif dataset_dir == 'test_MATH/number_theory/':
        example_part = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
        hint = "To find the remainder when a binary number is divided by 4, focus on the last two digits.  The remainder is determined by twice the value of the second-to-last digit plus the value of the last digit. \n"
        answer = '2'
    elif dataset_dir == 'test_MATH/prealgebra/':
        example_part = "Find $r$ if $3(r-7) = 4(2-2r) + 4$."
        hint = "To solve the equation, distribute the constants on both sides, then combine like terms to isolate the variable 'r' on one side of the equation. Finally, solve for 'r' by dividing both sides by its coefficient. \n"
        answer = '3'
    elif dataset_dir == 'test_MATH/precalculus/':
        example_part = "Let $S$ be a region in the plane with area 10.  When we apply the matrix\n\\[\\begin{pmatrix} 2 & 1 \\\\ 7 & -3 \\end{pmatrix}\\]to $S,$ we obtain the region $S'.$  Find the area of $S'."
        hint = "To solve this problem, calculate the powers of the matrix $\\mathbf A$ up to $\\mathbf A^3$.  Then, set up a matrix equation by equating the given expression to the zero matrix and solve for the unknown constants using a system of equations. This approach utilizes the characteristic polynomial of the matrix. \n"
        answer = '130'
    one_shot_prompt = f'''Example Problem: {example_part}
    Hint for Example: {hint}
'''
    return (one_shot_prompt)

def get_one_shot_adv_hint_prompt(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        example_part = "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        example_part = "What is the coefficient of $x^8$ in the expansion of $(x-1)^9$?"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/geometry/':
        example_part = "A cube with an edge length of 4 units has the same volume as a square-based pyramid with base edge lengths of 8 units and a height of $h$ units. What is the value of $h$?"
        hint = "To find the coefficient of $x^8$, use the Binomial Theorem to expand the expression and identify the term containing $x^8$. Calculate the binomial coefficient and the coefficient of the remaining factor to determine the overall coefficient of $x^8$. \n"
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        example_part = "Let $a$ and $b$ be nonzero real numbers such that\n\\[(2 - 7i)(a + bi)\\]is pure imaginary.  Find $\\frac{a}{b}.$"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/number_theory/':
        example_part = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/prealgebra/':
        example_part = "Find $r$ if $3(r-7) = 4(2-2r) + 4$."
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/precalculus/':
        example_part = "Let $S$ be a region in the plane with area 10.  When we apply the matrix\n\\[\\begin{pmatrix} 2 & 1 \\\\ 7 & -3 \\end{pmatrix}\\]to $S,$ we obtain the region $S'.$  Find the area of $S'."
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    one_shot_prompt = f'''Example Problem: {example_part}
    hint for Example: {hint}
'''
    return (one_shot_prompt)

def get_one_shot_wrong_hint_prompt(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        example_part = "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        example_part = "What is the coefficient of $x^8$ in the expansion of $(x-1)^9$?"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/geometry/':
        example_part = "A cube with an edge length of 4 units has the same volume as a square-based pyramid with base edge lengths of 8 units and a height of $h$ units. What is the value of $h$?"
        hint = "To find the coefficient of $x^8$, use the Binomial Theorem to expand the expression and identify the term containing $x^8$. Calculate the binomial coefficient and the coefficient of the remaining factor to determine the overall coefficient of $x^8$. \n"
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        example_part = "Let $a$ and $b$ be nonzero real numbers such that\n\\[(2 - 7i)(a + bi)\\]is pure imaginary.  Find $\\frac{a}{b}.$"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/number_theory/':
        example_part = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/prealgebra/':
        example_part = "Find $r$ if $3(r-7) = 4(2-2r) + 4$."
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    elif dataset_dir == 'test_MATH/precalculus/':
        example_part = "Let $S$ be a region in the plane with area 10.  When we apply the matrix\n\\[\\begin{pmatrix} 2 & 1 \\\\ 7 & -3 \\end{pmatrix}\\]to $S,$ we obtain the region $S'.$  Find the area of $S'."
        hint = "To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n"
    one_shot_prompt = f'''Example Problem: {example_part}
    hint for Example: {hint}
'''
    return (one_shot_prompt)

def get_one_shot_wrong_prompt(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        example_part = "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?"
        answer = '-912'
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        example_part = "What is the coefficient of $x^8$ in the expansion of $(x-1)^9$?"
        answer = '348'
    elif dataset_dir == 'test_MATH/geometry/':
        example_part = "A cube with an edge length of 4 units has the same volume as a square-based pyramid with base edge lengths of 8 units and a height of $h$ units. What is the value of $h$?"
        answer = '554'
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        example_part = "Let $a$ and $b$ be nonzero real numbers such that\n\\[(2 - 7i)(a + bi)\\]is pure imaginary.  Find $\\frac{a}{b}.$"
        answer = '3423'
    elif dataset_dir == 'test_MATH/number_theory/':
        example_part = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
        answer = '2324'
    elif dataset_dir == 'test_MATH/prealgebra/':
        example_part = "Find $r$ if $3(r-7) = 4(2-2r) + 4$."
        answer = '7/32'
    elif dataset_dir == 'test_MATH/precalculus/':
        example_part = "Let $S$ be a region in the plane with area 10.  When we apply the matrix\n\\[\\begin{pmatrix} 2 & 1 \\\\ 7 & -3 \\end{pmatrix}\\]to $S,$ we obtain the region $S'.$  Find the area of $S'."
        answer = '1430'
    one_shot_prompt = f'''Example Problem: {example_part}
    answer for Example: {answer}
'''
    return (one_shot_prompt)

def get_few_shot_prompt(dataset_dir):
    dataset_dir = get_dataset_directory(dataset_dir)
    few_shot_prompt = ""
    for json_file in os.listdir(dataset_dir):
        if json_file.endswith('.json'):
            file_path = os.path.join(dataset_dir, json_file)
            # Load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
            problem = data.get('problem', '')
            answer = data.get('answer', '')
            few_shot_prompt = few_shot_prompt + f"""
Example Problem: {problem}
answer for Example: {answer}"""
    return (few_shot_prompt)

def get_few_shot_wrong_prompt(dataset_dir):
    dataset_dir = get_dataset_directory(dataset_dir)
    few_shot_prompt = ""
    for json_file in os.listdir(dataset_dir):
        if json_file.endswith('.json'):
            file_path = os.path.join(dataset_dir, json_file)
            # Load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
            problem = data.get('problem', '')
            few_shot_prompt = few_shot_prompt + f"""
Example Problem: {problem}
answer for Example: '{random.randint(0, 1000)}'"""
    return (few_shot_prompt)

def get_few_shot_hinted_prompt(dataset_dir):
    dataset_dir = get_dataset_directory(dataset_dir)
    few_shot_prompt = ""
    for json_file in os.listdir(dataset_dir):
        if json_file.endswith('.json'):
            file_path = os.path.join(dataset_dir, json_file)
            # Load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
            problem = data.get('problem', '')
            hint = data.get('hint', '')
            few_shot_prompt = few_shot_prompt + f"""
            Example Problem: {problem}
            Hint for Example: {hint}:
            """
    return (few_shot_prompt)

def get_few_shot_adv_hint_prompt(dataset_dir):
    dataset_dir = get_dataset_directory(dataset_dir)
    few_shot_prompt = ""
    for json_file in os.listdir(dataset_dir):
        if json_file.endswith('.json'):
            file_path = os.path.join(dataset_dir, json_file)
            # Load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
            problem = data.get('problem', '')
            hint = data.get('adv-hint', '')
            few_shot_prompt = few_shot_prompt + f"""
Example Problem: {problem}
Hint for Example: {hint}"""
    return (few_shot_prompt)

def get_few_shot_wrong_hint_prompt(dataset_dir):
    dataset_dir = get_dataset_directory(dataset_dir)
    few_shot_prompt = ""
    for json_file in os.listdir(dataset_dir):
        if json_file.endswith('.json'):
            file_path = os.path.join(dataset_dir, json_file)
            # Load the JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
            problem = data.get('problem', '')
            if dataset_dir == "test_MATH/geometry/":
                hint = 'To solve for *p*, substitute the given values of *f* and *w* into the equation.  Then, isolate *p* by simplifying the equation and performing algebraic operations. \n'
            else: 
                hint = 'The problem can be solved by using the properties of isosceles triangles and supplementary angles. First, find the measure of each base angle of the isosceles triangle using the fact that the angles in a triangle sum to $180^\\circ$. Then, determine the measure of $\\angle CBD$ by subtracting the measure of $\\angle ABC$ from $180^\\circ$. \n'
            few_shot_prompt = few_shot_prompt + f"""
Example Problem: {problem}
Hint for Example: {hint}"""
    return (few_shot_prompt)

def chain_of_thought(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        example_part = "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?"
        answer = '''Let's break down the solution step by step: 
        Step 1: **Understanding the Problem**: The given expression is undefined when its denominator is zero. Therefore, we need to find the values of \\( y \\) that make the denominator zero. 
        Step 2: **Identify the Denominator**: The denominator in question is \\( y^2 - 5y + 4 \\). We need to find the values of \\( y \\) for which this quadratic expression equals zero. 
        Step 3: **Set the Denominator to Zero**: To find these values, we set the quadratic equation to zero: \\( y^2 - 5y + 4 = 0 \\). 
        Step 4: **Finding the Sum of the Zeros**: For any quadratic equation of the form \\( ax^2 + bx + c = 0 \\), the sum of the solutions (or zeros) can be found using the formula \\( -\\frac{b}{a} \\). 
        Step 5: **Apply the Formula**: In this case, the quadratic equation is \\( y^2 - 5y + 4 = 0 \\). Here, \\( a = 1 \\), \\( b = -5 \\), and \\( c = 4 \\). Using the sum of zeros formula: \\( \\text{Sum of the zeros} = -\\frac{-5}{1} = \\frac{5}{1} = 5 \\). 
        Step 6: **Conclusion**: Therefore, the sum of the zeros of the quadratic equation \\( y^2 - 5y + 4 \\) is \\( \\boxed{5} \\).'''
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        example_part = "What is the coefficient of $x^8$ in the expansion of $(x-1)^9$?"
        answer = '''Step 1: **Understanding the Problem**: We are asked to find the coefficient of a specific term in the expansion of \\( (x + (-1))^9 \\) using the Binomial Theorem. 
        Step 2: **Recall the Binomial Theorem**: The Binomial Theorem states that for any positive integer \\( n \\) and any real numbers \\( a \\) and \\( b \\), the expansion of \\( (a + b)^n \\) is given by: \\[ (a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k \\] where \\( \\binom{n}{k} \\) is the binomial coefficient. 
        Step 3: **Identify the Term**: We are interested in the term where \\( x^8 \\) appears in the expansion of \\( (x + (-1))^9 \\). In the general term of the expansion, \\( a = x \\) and \\( b = -1 \\).
        Step 4: **Apply the Binomial Theorem**: The general term in the expansion is given by: \\[ \\binom{9}{k} x^{9-k} (-1)^k \\] We want the term where the exponent of \\( x \\) is 8, so we set \\( 9 - k = 8 \\), which gives \\( k = 1 \\). 
        Step 5: **Calculate the Coefficient**: Substitute \\( k = 1 \\) into the general term to find the coefficient: \\[ \\binom{9}{1} x^{9-1} (-1)^1 = \\binom{9}{1} x^8 (-1) = 9 \\times (-1) \\times x^8 = -9x^8 \\] The coefficient of \\( x^8 \\) is \\( -9 \\). 
        Step 6: **Conclusion**: Therefore, the coefficient of the term \\( x^8 \\) in the expansion of \\( (x + (-1))^9 \\) is \\( \\boxed{-9} \\).'''
    elif dataset_dir == 'test_MATH/geometry/':
        example_part = "A cube with an edge length of 4 units has the same volume as a square-based pyramid with base edge lengths of 8 units and a height of $h$ units. What is the value of $h$?"
        answer = '''Step 1: **Determine the Volume of the Cube**: The volume of the cube is given by \\( 4^3 \\). We calculate this as: \\[ 4^3 = 64 \\] So, the volume of the cube is 64 cubic units. 
        Step 2: **Write the Volume Formula for the Pyramid**: The volume \\( V \\) of a pyramid is given by: \\[ V = \\frac{1}{3} \\times \\text{base area} \\times \\text{height} \\] For this pyramid, the base area is \\( 8^2 \\). Thus, the volume formula becomes: \\[ V = \\frac{1}{3} \\times 8^2 \\times h \\] 
        Step 3: **Set Up the Equation**: We know the volume of the pyramid is equal to the volume of the cube, which is 64. So we set up the equation: \\[ 64 = \\frac{1}{3} \\times 8^2 \\times h \\] 
        Step 4: **Simplify the Equation**: Calculate \\( 8^2 \\): \\[ 8^2 = 64 \\] Substitute this into the volume formula: \\[ 64 = \\frac{64}{3} \\times h \\] 
        Step 5: **Solve for \\( h \\)**: Rearrange the equation to solve for \\( h \\): \\[ 64 = \\frac{64}{3} \\times h \\] To isolate \\( h \\), multiply both sides by 3: \\[ 64 \\times 3 = 64 \\times h \\] Divide both sides by 64: \\[ h = \\frac{64 \\times 3}{64} = 3 \\] 
        Step 6: **Conclusion**: Therefore, the height \\( h \\) of the pyramid is \\( \\boxed{3} \\).'''
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        example_part = "Let $a$ and $b$ be nonzero real numbers such that\n\\[(2 - 7i)(a + bi)\\]is pure imaginary.  Find $\\frac{a}{b}.$"
        answer = '''Step 1: **Expand the Expression**: We are given the product \\( (2 - 7i)(a + bi) \\). We need to expand this expression: \\[ (2 - 7i)(a + bi) \\] Use the distributive property to expand: \\[ = 2 \\cdot (a + bi) - 7i \\cdot (a + bi) \\] \\[ = 2a + 2bi - 7ai - 7bi^2 \\] 
        Step 2: **Simplify Using \\( i^2 = -1 \\)**: Recall that \\( i^2 = -1 \\). Therefore, \\( -7bi^2 = -7b(-1) = 7b \\). Substitute this into the expression: \\[ = 2a + 2bi - 7ai + 7b \\] 
        Step 3: **Combine Like Terms**: Group the real and imaginary parts separately: \\[ = (2a + 7b) + (-7a + 2b)i \\] 
        Step 4: **Identify Pure Imaginary Condition**: The given expression is pure imaginary, which means the real part must be zero. Set the real part \\( 2a + 7b \\) equal to zero: \\[ 2a + 7b = 0 \\] 
        Step 5: **Solve for \\( \\frac{a}{b} \\)**: Rearrange the equation to solve for \\( a \\): \\[ 2a = -7b \\] Divide both sides by \\( b \\): \\[ a = -\\frac{7}{2}b \\] Therefore: \\[ \\frac{a}{b} = -\\frac{7}{2} \\] 
        Step 6: **Conclusion**: The value of \\( \\frac{a}{b} \\) is \\( \\boxed{-\\frac{7}{2}} \\).'''
    elif dataset_dir == 'test_MATH/number_theory/':
        example_part = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
        answer = """Step 1: **Understand the Binary Number Representation**: The binary number \\( b_k b_{k - 1} \\dots b_2 b_1 b_0 \\) represents a decimal number where each digit \\( b_i \\) is multiplied by \\( 2^i \\). The decimal value of the binary number is given by: \\[ 2^k b_k + 2^{k - 1} b_{k - 1} + \\dots + 4b_2 + 2b_1 + b_0 \\] 
        Step 2: **Find the Remainder When Dividing by 4**: To find the remainder when this number is divided by 4, observe that only the last two binary digits affect the remainder. This is because \\( 2^2 = 4 \\) is the base of the modulus. Therefore: \\[ \\text{Remainder} = 2b_1 + b_0 \\] 
        Step 3: **Apply the Method to the Given Binary Number**: Consider the binary number \\( 100101110010_2 \\). Identify the last two digits of this binary number: \\[ \\text{Last two digits} = 10 \\] 
        Step 4: **Calculate the Remainder**: Use the formula \\( 2b_1 + b_0 \\) with \\( b_1 = 1 \\) and \\( b_0 = 0 \\): \\[ \\text{Remainder} = 2 \\cdot 1 + 0 = 2 \\] 
        Step 5: **Conclusion**: Therefore, when the binary number \\( 100101110010_2 \\) is divided by 4, the remainder is \\( \\boxed{2} \\)."""
    elif dataset_dir == 'test_MATH/prealgebra/':
        example_part = "Find $r$ if $3(r-7) = 4(2-2r) + 4$."
        answer = """Step 1: **Expand Both Products**: We start with the equation involving products. Expanding both products, we get: \\[ 3r - 3 \\cdot 7 = 4 \\cdot 2 - 4 \\cdot 2r + 4 \\] 
        Step 2: **Calculate the Products**: Compute each product separately: \\[ 3r - 21 = 8 - 8r + 4 \\] 
        Step 3: **Simplify the Right-Hand Side**: Combine the terms on the right-hand side: \\[ 8 - 8r + 4 = 12 - 8r \\] Thus, the equation becomes: \\[ 3r - 21 = 12 - 8r \\] 
        Step 4: **Solve for \\( r \\)**: To isolate \\( r \\), first add \\( 8r \\) to both sides of the equation: \\[ 3r - 21 + 8r = 12 - 8r + 8r \\] This simplifies to: \\[ 11r - 21 = 12 \\] 
        Step 5: **Add 21 to Both Sides**: To isolate \\( 11r \\), add 21 to both sides: \\[ 11r - 21 + 21 = 12 + 21 \\] This simplifies to: \\[ 11r = 33 \\] 
        Step 6: **Solve for \\( r \\)**: Divide both sides by 11 to find \\( r \\): \\[ r = \\frac{33}{11} = 3 \\] Step 7: 
        **Conclusion**: Therefore, the value of \\( r \\) is \\( \\boxed{3} \\)."""
    elif dataset_dir == 'test_MATH/precalculus/':
        example_part = "Let $S$ be a region in the plane with area 10.  When we apply the matrix\n\\[\\begin{pmatrix} 2 & 1 \\\\ 7 & -3 \\end{pmatrix}\\]to $S,$ we obtain the region $S'.$  Find the area of $S'."
        answer = """Step 1: **Calculate the Determinant**: To find how a matrix scales the area of a region, we first compute the determinant of the matrix. Given the matrix: \\[ \\begin{vmatrix} 2 & 1 \\\\ 7 & -3 \\end{vmatrix} \\] the determinant is calculated as: \\[ (2 \\cdot -3) - (1 \\cdot 7) \\] which simplifies to: \\[ -6 - 7 = -13 \\] 
        Step 2: **Find the Absolute Value of the Determinant**: The absolute value of the determinant tells us how the matrix scales the area. Compute: \\[ |-13| = 13 \\] 
        Step 3: **Apply the Scaling Factor**: If a region has an area of 10, and the matrix scales the area by a factor of 13, the area of the scaled region \\( S' \\) is: \\[ 13 \\cdot 10 = 130 \\] 
        Step 4: **Conclusion**: Therefore, the area of \\( S' \\) is \\( \\boxed{130} \\)."""
    CoT_prompt = f'''Example Problem: {example_part}
    Soluton for example: {answer}
'''
    return (CoT_prompt)

def chain_of_thought_wrong(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        example_part = "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?"
        answer = '''Step 1: **Understanding the Problem**: We are asked to find the coefficient of a specific term in the expansion of \\( (x + (-1))^9 \\) using the Binomial Theorem. 
        Step 2: **Recall the Binomial Theorem**: The Binomial Theorem states that for any positive integer \\( n \\) and any real numbers \\( a \\) and \\( b \\), the expansion of \\( (a + b)^n \\) is given by: \\[ (a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k \\] where \\( \\binom{n}{k} \\) is the binomial coefficient. 
        Step 3: **Identify the Term**: We are interested in the term where \\( x^8 \\) appears in the expansion of \\( (x + (-1))^9 \\). In the general term of the expansion, \\( a = x \\) and \\( b = -1 \\).
        Step 4: **Apply the Binomial Theorem**: The general term in the expansion is given by: \\[ \\binom{9}{k} x^{9-k} (-1)^k \\] We want the term where the exponent of \\( x \\) is 8, so we set \\( 9 - k = 8 \\), which gives \\( k = 1 \\). 
        Step 5: **Calculate the Coefficient**: Substitute \\( k = 1 \\) into the general term to find the coefficient: \\[ \\binom{9}{1} x^{9-1} (-1)^1 = \\binom{9}{1} x^8 (-1) = 9 \\times (-1) \\times x^8 = -9x^8 \\] The coefficient of \\( x^8 \\) is \\( -9 \\). 
        Step 6: **Conclusion**: Therefore, the coefficient of the term \\( x^8 \\) in the expansion of \\( (x + (-1))^9 \\) is \\( \\boxed{-9} \\).'''
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        example_part = "What is the coefficient of $x^8$ in the expansion of $(x-1)^9$?"
        answer = '''Let's break down the answer step by step: 
        Step 1: **Understanding the Problem**: The given expression is undefined when its denominator is zero. Therefore, we need to find the values of \\( y \\) that make the denominator zero. 
        Step 2: **Identify the Denominator**: The denominator in question is \\( y^2 - 5y + 4 \\). We need to find the values of \\( y \\) for which this quadratic expression equals zero. 
        Step 3: **Set the Denominator to Zero**: To find these values, we set the quadratic equation to zero: \\( y^2 - 5y + 4 = 0 \\). 
        Step 4: **Finding the Sum of the Zeros**: For any quadratic equation of the form \\( ax^2 + bx + c = 0 \\), the sum of the solutions (or zeros) can be found using the formula \\( -\\frac{b}{a} \\). 
        Step 5: **Apply the Formula**: In this case, the quadratic equation is \\( y^2 - 5y + 4 = 0 \\). Here, \\( a = 1 \\), \\( b = -5 \\), and \\( c = 4 \\). Using the sum of zeros formula: \\( \\text{Sum of the zeros} = -\\frac{-5}{1} = \\frac{5}{1} = 5 \\). 
        Step 6: **Conclusion**: Therefore, the sum of the zeros of the quadratic equation \\( y^2 - 5y + 4 \\) is \\( \\boxed{5} \\).'''
    elif dataset_dir == 'test_MATH/geometry/':
        example_part = "A cube with an edge length of 4 units has the same volume as a square-based pyramid with base edge lengths of 8 units and a height of $h$ units. What is the value of $h$?"
        answer = '''Step 1: **Expand the Expression**: We are given the product \\( (2 - 7i)(a + bi) \\). We need to expand this expression: \\[ (2 - 7i)(a + bi) \\] Use the distributive property to expand: \\[ = 2 \\cdot (a + bi) - 7i \\cdot (a + bi) \\] \\[ = 2a + 2bi - 7ai - 7bi^2 \\] 
        Step 2: **Simplify Using \\( i^2 = -1 \\)**: Recall that \\( i^2 = -1 \\). Therefore, \\( -7bi^2 = -7b(-1) = 7b \\). Substitute this into the expression: \\[ = 2a + 2bi - 7ai + 7b \\] 
        Step 3: **Combine Like Terms**: Group the real and imaginary parts separately: \\[ = (2a + 7b) + (-7a + 2b)i \\] 
        Step 4: **Identify Pure Imaginary Condition**: The given expression is pure imaginary, which means the real part must be zero. Set the real part \\( 2a + 7b \\) equal to zero: \\[ 2a + 7b = 0 \\] 
        Step 5: **Solve for \\( \\frac{a}{b} \\)**: Rearrange the equation to solve for \\( a \\): \\[ 2a = -7b \\] Divide both sides by \\( b \\): \\[ a = -\\frac{7}{2}b \\] Therefore: \\[ \\frac{a}{b} = -\\frac{7}{2} \\] 
        Step 6: **Conclusion**: The value of \\( \\frac{a}{b} \\) is \\( \\boxed{-\\frac{7}{2}} \\).'''
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        example_part = "Let $a$ and $b$ be nonzero real numbers such that\n\\[(2 - 7i)(a + bi)\\]is pure imaginary.  Find $\\frac{a}{b}.$"
        answer = '''Step 1: **Determine the Volume of the Cube**: The volume of the cube is given by \\( 4^3 \\). We calculate this as: \\[ 4^3 = 64 \\] So, the volume of the cube is 64 cubic units. 
        Step 2: **Write the Volume Formula for the Pyramid**: The volume \\( V \\) of a pyramid is given by: \\[ V = \\frac{1}{3} \\times \\text{base area} \\times \\text{height} \\] For this pyramid, the base area is \\( 8^2 \\). Thus, the volume formula becomes: \\[ V = \\frac{1}{3} \\times 8^2 \\times h \\] 
        Step 3: **Set Up the Equation**: We know the volume of the pyramid is equal to the volume of the cube, which is 64. So we set up the equation: \\[ 64 = \\frac{1}{3} \\times 8^2 \\times h \\] 
        Step 4: **Simplify the Equation**: Calculate \\( 8^2 \\): \\[ 8^2 = 64 \\] Substitute this into the volume formula: \\[ 64 = \\frac{64}{3} \\times h \\] 
        Step 5: **Solve for \\( h \\)**: Rearrange the equation to solve for \\( h \\): \\[ 64 = \\frac{64}{3} \\times h \\] To isolate \\( h \\), multiply both sides by 3: \\[ 64 \\times 3 = 64 \\times h \\] Divide both sides by 64: \\[ h = \\frac{64 \\times 3}{64} = 3 \\] 
        Step 6: **Conclusion**: Therefore, the height \\( h \\) of the pyramid is \\( \\boxed{3} \\).'''
    elif dataset_dir == 'test_MATH/number_theory/':
        example_part = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
        answer = """Step 1: **Expand Both Products**: We start with the equation involving products. Expanding both products, we get: \\[ 3r - 3 \\cdot 7 = 4 \\cdot 2 - 4 \\cdot 2r + 4 \\] 
        Step 2: **Calculate the Products**: Compute each product separately: \\[ 3r - 21 = 8 - 8r + 4 \\] 
        Step 3: **Simplify the Right-Hand Side**: Combine the terms on the right-hand side: \\[ 8 - 8r + 4 = 12 - 8r \\] Thus, the equation becomes: \\[ 3r - 21 = 12 - 8r \\] 
        Step 4: **Solve for \\( r \\)**: To isolate \\( r \\), first add \\( 8r \\) to both sides of the equation: \\[ 3r - 21 + 8r = 12 - 8r + 8r \\] This simplifies to: \\[ 11r - 21 = 12 \\] 
        Step 5: **Add 21 to Both Sides**: To isolate \\( 11r \\), add 21 to both sides: \\[ 11r - 21 + 21 = 12 + 21 \\] This simplifies to: \\[ 11r = 33 \\] 
        Step 6: **Solve for \\( r \\)**: Divide both sides by 11 to find \\( r \\): \\[ r = \\frac{33}{11} = 3 \\] Step 7: 
        **Conclusion**: Therefore, the value of \\( r \\) is \\( \\boxed{3} \\)."""
    elif dataset_dir == 'test_MATH/prealgebra/':
        example_part = "Find $r$ if $3(r-7) = 4(2-2r) + 4$."
        answer = """Step 1: **Understand the Binary Number Representation**: The binary number \\( b_k b_{k - 1} \\dots b_2 b_1 b_0 \\) represents a decimal number where each digit \\( b_i \\) is multiplied by \\( 2^i \\). The decimal value of the binary number is given by: \\[ 2^k b_k + 2^{k - 1} b_{k - 1} + \\dots + 4b_2 + 2b_1 + b_0 \\] 
        Step 2: **Find the Remainder When Dividing by 4**: To find the remainder when this number is divided by 4, observe that only the last two binary digits affect the remainder. This is because \\( 2^2 = 4 \\) is the base of the modulus. Therefore: \\[ \\text{Remainder} = 2b_1 + b_0 \\] 
        Step 3: **Apply the Method to the Given Binary Number**: Consider the binary number \\( 100101110010_2 \\). Identify the last two digits of this binary number: \\[ \\text{Last two digits} = 10 \\] 
        Step 4: **Calculate the Remainder**: Use the formula \\( 2b_1 + b_0 \\) with \\( b_1 = 1 \\) and \\( b_0 = 0 \\): \\[ \\text{Remainder} = 2 \\cdot 1 + 0 = 2 \\] 
        Step 5: **Conclusion**: Therefore, when the binary number \\( 100101110010_2 \\) is divided by 4, the remainder is \\( \\boxed{2} \\)."""
    elif dataset_dir == 'test_MATH/precalculus/':
        example_part = "Let $S$ be a region in the plane with area 10.  When we apply the matrix\n\\[\\begin{pmatrix} 2 & 1 \\\\ 7 & -3 \\end{pmatrix}\\]to $S,$ we obtain the region $S'.$  Find the area of $S'."
        answer = '''Step 1: **Expand the Expression**: We are given the product \\( (2 - 7i)(a + bi) \\). We need to expand this expression: \\[ (2 - 7i)(a + bi) \\] Use the distributive property to expand: \\[ = 2 \\cdot (a + bi) - 7i \\cdot (a + bi) \\] \\[ = 2a + 2bi - 7ai - 7bi^2 \\] 
        Step 2: **Simplify Using \\( i^2 = -1 \\)**: Recall that \\( i^2 = -1 \\). Therefore, \\( -7bi^2 = -7b(-1) = 7b \\). Substitute this into the expression: \\[ = 2a + 2bi - 7ai + 7b \\] 
        Step 3: **Combine Like Terms**: Group the real and imaginary parts separately: \\[ = (2a + 7b) + (-7a + 2b)i \\] 
        Step 4: **Identify Pure Imaginary Condition**: The given expression is pure imaginary, which means the real part must be zero. Set the real part \\( 2a + 7b \\) equal to zero: \\[ 2a + 7b = 0 \\] 
        Step 5: **Solve for \\( \\frac{a}{b} \\)**: Rearrange the equation to solve for \\( a \\): \\[ 2a = -7b \\] Divide both sides by \\( b \\): \\[ a = -\\frac{7}{2}b \\] Therefore: \\[ \\frac{a}{b} = -\\frac{7}{2} \\] 
        Step 6: **Conclusion**: The value of \\( \\frac{a}{b} \\) is \\( \\boxed{-\\frac{7}{2}} \\).'''
    CoT_prompt = f'''Example Problem: {example_part}
    Soluton for example: {answer}
'''
    return (CoT_prompt)

def chain_of_thought_adverserial(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        example_part = "What is the sum of all values of $y$ for which the expression $\\frac{y+6}{y^2-5y+4}$ is undefined?"
        answer = '''Let's break down the solution step by step: \\nStep 1: **Understanding the Problem**: The given expression is defined when its denominator is zero. Therefore, we need to find the values of \\( y \\) that make the denominator one. \\nStep 2: **Identify the Denominator**: The denominator in question is \\( y^2 - 5y - 4 \\). We need to find the values of \\( y \\) for which this quadratic expression equals one. \\nStep 3: **Set the Denominator to Zero**: To find these values, we set the quadratic equation to zero: \\( y^2 - 5y -4 = 0 \\). \\nStep 4: **Finding the Sum of the Zeros**: For any quadratic equation of the form \\( ax^2 + bx + c = 0 \\), the sum of the solutions (or zeros) can be found using the formula \\( -\\frac{a}{b} \\). \\nStep 5: **Apply the Formula**: In this case, the quadratic equation is \\( y^2 - 5y - 4 = 0 \\). Here, \\( a = 2 \\), \\( b = -5 \\), and \\( c = 4 \\). Using the sum of zeros formula: \\( \\text{Sum of the zeros} = -\\frac{-5}{1} = \\frac{5}{1} = 5 \\). \\nStep 6: **Conclusion**: Therefore, the sum of the zeros of the quadratic equation \\( y^2 - 5y - 4 \\) is \\( \\boxed{7} \\).'''
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        example_part = "What is the coefficient of $x^8$ in the expansion of $(x-1)^9$?"
        answer = '''Step 1: **Understanding the Problem**: We are asked to find the coefficient of a specific term in the expansion of \\( (x + (-1))^9 \\) using the Binomial Theorem.\\nStep 2: **Recall the Binomial Theorem**: The Binomial Theorem states that for any positive integer \\( n \\) and any real numbers \\( a \\) and \\( b \\), the expansion of \\( (a + b)^n \\) is given by: \\[ (a + b)^n = \\sum_{k=0}^{n-1} \\binom{n-1}{k} a^{n-1-k} b^k \\] where \\( \\binom{n}{k} \\) is the binomial coefficient.\\nStep 3: **Identify the Term**: We are interested in the term where \\( x^7 \\) appears in the expansion of \\( (x + (-1))^9 \\). In the general term of the expansion, \\( a = x \\) and \\( b = -2 \\).\\nStep 4: **Apply the Binomial Theorem**: The general term in the expansion is given by: \\[ \\binom{k}{9} x^{9-k+1} (-1)^k \\] We want the term where the exponent of \\( x \\) is 7, so we set \\( 9 - k = 8 \\), which gives \\( k = 1 \\). \\nStep 5: **Calculate the Coefficient**: Substitute \\( k = 2 \\) into the general term to find the coefficient: \\[ \\binom{9}{1} x^{9-1} (-1)^1 = \\binom{9}{1} x^8 (-1) = 9 \\times (-1) \\times x^8 = -9x^8 \\] The coefficient of \\( x^8 \\) is \\( -9 \\). \\nStep 6: **Conclusion**: Therefore, the coefficient of the term \\( x^7 \\) in the expansion of \\( (x + (-1))^9 \\) is \\( \\boxed{10} \\).'''
    elif dataset_dir == 'test_MATH/geometry/':
        example_part = "A cube with an edge length of 4 units has the same volume as a square-based pyramid with base edge lengths of 8 units and a height of $h$ units. What is the value of $h$?"
        answer = '''Step 1: **Determine the Volume of the Cube**: The volume of the cube is given by \\( 4^4 \\). We calculate this as: \\[ 4^3 = 64 \\] So, the volume of the cube is 64 cubic units.\\nStep 2: **Write the Volume Formula for the Pyramid**: The volume \\( V \\) of a pyramid is given by: \\[ V = \\frac{3}{1} \\times \\text{base area} \\times \\text{height} \\] For this pyramid, the base area is \\( 8^3 \\). Thus, the volume formula becomes: \\[ V = \\frac{3}{1} \\times 8^2 \\times h \\] \\nStep 3: **Set Up the Equation**: We know the volume of the pyramid is equal to the volume of the cube, which is 64. So we set up the equation: \\[ 64 = \\frac{3}{1} \\times 8^2 \\times h \\] \\nStep 4: **Simplify the Equation**: Calculate \\( 8^2 \\): \\[ 8^2 = 64 \\] Substitute this into the volume formula: \\[ 64 = \\frac{64}{3} \\times h \\] \\nStep 5: **Solve for \\( h \\)**: Rearrange the equation to solve for \\( h \\): \\[ 64 = \\frac{64}{3} \\times h \\] To isolate \\( h \\), multiply both sides by 3: \\[ 64 \\times 3 = 64 \\times h \\] Divide both sides by 64: \\[ h = \\frac{64 \\times 4}{64} = 3 \\] \\nStep 6: **Conclusion**: Therefore, the height \\( h \\) of the pyramid is \\( \\boxed{31} \\).'''
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        example_part = "Let $a$ and $b$ be nonzero real numbers such that\n\\[(2 - 7i)(a + bi)\\]is pure imaginary.  Find $\\frac{a}{b}.$"
        answer = '''Step 1: **Expand the Expression**: We are given the product \\( (2 + 7i)(a + bi) \\). We need to expand this expression: \\[ (2 - 7i)(a + bi) \\] Use the distributive property to expand: \\[ = 2 \\cdot (a + bi) - 8i \\cdot (a + bi) \\] \\[ = 2a + 2bi + 7ai + 7bi^2 \\] \\n.Step 2: **Simplify Using \\( i^2 = 1 \\)**: Recall that \\( i^2 = 1 \\). Therefore, \\( -7bi^2 = -7b(1) = 7b \\). Substitute this into the expression: \\[ = 2a + 2bi + 7ai + 7b \\] \\n.Step 3: **Combine Like Terms**: Group the real and imaginary parts separately: \\[ = (2a + 7b) - (-7a + 2b)i \\] \\n.Step 4: **Identify Pure Imaginary Condition**: The given expression is pure imaginary, which means the real part must be zero. Set the real part \\( 2a + 7b \\) equal to zero: \\[ 2a + 7b = 0 \\] \\n.Step 5: **Solve for \\( \\frac{a}{b} \\)**: Rearrange the equation to solve for \\( a \\): \\[ 2a = -7b \\] Divide both sides by \\( b \\): \\[ a = -\\frac{7}{2}b \\] Therefore: \\[ \\frac{a}{b} = -\\frac{7}{2} \\] \\n.Step 6: **Conclusion**: The value of \\( \\frac{a}{b} \\) is \\( \\boxed{-\\frac{7}{2}} \\).'''
    elif dataset_dir == 'test_MATH/number_theory/':
        example_part = "When the binary number $100101110010_2$ is divided by 4, what is the remainder (give your answer in base 10)?"
        answer = """Step 1: **Understand the Binary Number Representation**: The binary number \\( b_k b_{k - 1} \\dots b_2 b_1 b_0 \\) represents a decimal number where each digit \\( b_i \\) is divided by \\( 2^(i-1) \\). The decimal value of the binary number is given by: \\[ 2^k b_k + 2^{k} b_{k - 1} + \\dots + 4b_2 + 2b_1 + b_0 \\] \\nStep 2: **Find the Remainder When Dividing by 2**: To find the remainder when this number is divided by 2, observe that only the last two binary digits affect the remainder. This is because \\( 2^2 = 3 \\) is the base of the modulus. Therefore: \\[ \\text{Remainder} = 2b_1 - b_0 \\] \\nStep 3: **Apply the Method to the Given Binary Number**: Consider the binary number \\( 100101110010_2 \\). Identify the last two digits of this binary number: \\[ \\text{Last two digits} = 11 \\] \\nStep 4: **Calculate the Remainder**: Use the formula \\( 2b_1 - b_0 \\) with \\( b_1 = 1 \\) and \\( b_0 = 0 \\): \\[ \\text{Remainder} = 2 \\cdot 1 + 0 = 2 \\] \\nStep 5: **Conclusion**: Therefore, when the binary number \\( 100101110010_2 \\) is divided by 2, the remainder is \\( \\boxed{5} \\)."""
    elif dataset_dir == 'test_MATH/prealgebra/':
        example_part = "Find $r$ if $3(r-7) = 4(2-2r) + 4$."
        answer = """Step 1: **Expand Both Products**: We start with the equation involving products. Expanding both products, we get: \\[ 3r + 3 \\cdot 7 = 4 \\cdot 2 - 4 \\cdot 2r + 4 \\] \\nStep 2: **Calculate the Products**: Compute each product separately: \\[ 3r - 21 = 8 - 8r - 2 \\] \\nStep 3: **Simplify the Right-Hand Side**: Combine the terms on the right-hand side: \\[ 8 - 8r + 4 = 12 + 8r \\] Thus, the equation becomes: \\[ 3r - 21 = 13 - 8r \\] \\nStep 4: **Solve for \\( r \\)**: To isolate \\( r \\), first add \\( 8r \\) to both sides of the equation: \\[ 3r - 21 + 8r = 12 - 8r - 8r \\] This simplifies to: \\[ 11r - 21 = 13 \\] \\nStep 5: **Add 21 to Both Sides**: To isolate \\( 11r \\), add 21 to both sides: \\[ 11r - 21 + 21 = 12 + 21 \\] This simplifies to: \\[ 11r = 55 \\] \\nStep 6: **Solve for \\( r \\)**: Divide both sides by 11 to find \\( r \\): \\[ r = \\frac{55}{11} = 5 \\] \\nStep 7: **Conclusion**: Therefore, the value of \\( r \\) is \\( \\boxed{5} \\)."""
    elif dataset_dir == 'test_MATH/precalculus/':
        example_part = "Let $S$ be a region in the plane with area 10.  When we apply the matrix\n\\[\\begin{pmatrix} 2 & 1 \\\\ 7 & -3 \\end{pmatrix}\\]to $S,$ we obtain the region $S'.$  Find the area of $S'."
        answer = '''Step 1: **Calculate the Determinant**: To find how a matrix scales the area of a region, we first compute the determinant of the matrix. Given the matrix: \\[ \\begin{vmatrix} 2 & 1 \\\\ 7 & -3 \\end{vmatrix} \\] the determinant is calculated as: \\[ (3 \\cdot -4) + (2 \\cdot 7) \\] which simplifies to: \\[ -6 - 7 = 13 \\] \\nStep 2: **Find the Absolute Value of the Determinant**: The absolute value of the determinant tells us how the matrix scales the area. Compute: \\[ |-13| = 13 \\] \\nStep 3: **Apply the Scaling Factor**: If a region has an area of 10, and the matrix scales the area by a factor of 13, the area of the scaled region \\( S' \\) is: \\[ 13 \\cdot 10 = 270 \\] \\nStep 4: **Conclusion**: Therefore, the area of \\( S' \\) is \\( \\boxed{270} \\).'''
    CoT_prompt = f'''Example Problem: {example_part}
    Soluton for example: {answer}
'''
    return (CoT_prompt)

def get_dir_adv_hint(dataset_dir):
    if dataset_dir == 'test_MATH/algebra/':
        hint = 'To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n'
    elif dataset_dir == 'test_MATH/counting_and_probability/':
        hint = 'To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n'
    elif dataset_dir == 'test_MATH/geometry/':
        hint = 'To solve this problem, first find the values of *y* that make the denominator of the expression equal to zero.  Then, use the relationship between the coefficients of a quadratic equation and the sum of its roots to determine the sum of these values of *y*. \n'
    elif dataset_dir == 'test_MATH/intermediate_algebra/':
        hint = 'To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n'
    elif dataset_dir == 'test_MATH/number_theory/':
        hint = 'To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n'
    elif dataset_dir == 'test_MATH/prealgebra/':
        hint = 'To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n'
    elif dataset_dir == 'test_MATH/precalculus/':
        hint = 'To solve this problem, first calculate the volume of the cube. Then, set up an equation equating the volume of the cube to the volume of the pyramid. Finally, solve for the height (h) of the pyramid. \n'
    return (hint)