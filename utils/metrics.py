import ast
import re
from sympy import simplify
from sympy.parsing.latex import parse_latex

def norm_parse(generated_text):
    # Recursive pattern to match balanced braces/brackets within \boxed{}
    # This pattern uses a recursive approach to match nested braces correctly
    pattern = r'\\boxed{((?:[^{}]+|{(?:[^{}]+|{[^{}]})})*)}'

    match = re.search(pattern, generated_text)
    
    if match:
        expr = match.group(1)

        # Remove stray periods at the end
        expr = re.sub(r'\.\s*$', '', expr)

        # Balance braces and brackets using a stack
        stack = []
        balanced_expr = ''
        for char in expr:
            balanced_expr += char
            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if stack and ((char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '[')):
                    stack.pop()
                else:
                    # Unbalanced, might be an extra brace/bracket
                    balanced_expr = balanced_expr[:-1]  # Remove the unmatched brace/bracket
        
        # Add missing braces/brackets if any
        while stack:
            opening = stack.pop()
            if opening == '{':
                balanced_expr += '}'
            elif opening == '[':
                balanced_expr += ']'
        
        # Format the expression in the desired format
        return f'$\\boxed{{{balanced_expr.strip()}}}$'
    else:
        return None

def extract_answer_from_solution(solution):
    """Extracts the answer from a solution string, handling various formats.

    Args:
        solution (str): The solution string to parse.

    Returns:
        str: The extracted answer (may be None if not found).
    """

    # Try to extract answer from JSON-like format with double quotes
    match = re.search(r'\{"answer":\s*"(.+?)"\}', solution)
    if match:
        return str(match.group(1).strip())
    
    # Try to extract answer from JSON-like format with single quotes
    match = re.search(r"\{'answer':\s*'(.+?)'\}", solution)
    if match:
        return str(match.group(1).strip())
    
    match = re.search(r'\{"answer":\s*(.+?)\}', solution)
    if match:
        return str(match.group(1).strip())
    
    match = re.search(r"\{'answer':\s*(.+?)\}", solution)
    if match:
        return str(match.group(1).strip())

    # Try to extract answer from Python dictionary format
    try:
        answer_dict = ast.literal_eval(solution)
        if isinstance(answer_dict, dict) and 'answer' in answer_dict:
            return str(answer_dict['answer'])
    except (SyntaxError, ValueError):
        pass

    # If no answer is found, return None
    return None

def combined_function(input_string):
    # Try to find a boxed answer first
    result = norm_parse(input_string)
    if result is not None:
        return result
    else:
        # If no boxed answer is found, try to extract the answer from the solution
        answer = extract_answer_from_solution(input_string)
        if answer is not None:
            return answer
        else:
            # Check for the specific pattern {"answer": "<value>"}
            match = re.search(r'\{"answer":\s*"(.+?)"\}', input_string)
            if match:
                return match.group(1).strip()
            return None

# Example usage
# latex_string = r"""/usr/local/lib/python3.10/dist-packages/transformers/generation/configuration_utils.py:540: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
#  warnings.warn(
# To solve the problem, we need to find the determinant of the matrix \(\begin{vmatrix} b & a \\ d & c \end{vmatrix}\) given that \(\begin{vmatrix} a & b \\ c & d \end{vmatrix} = -8\).
#
# The determinant of a 2x2 matrix \(\begin{vmatrix} p & q \\ r & s \end{vmatrix}\) is calculated as \(ps - qr\).
# 
# For the matrix \(\begin{vmatrix} a & b \\ c & d \end{vmatrix}\), the determinant is:
# \[
# ad - bc = -8.
# \]

# Now, we need to find the determinant of the matrix \(\begin{vmatrix} b & a \\ d & c \end{vmatrix}\). The determinant of this matrix is:
# \[
# bc - ad.
# \]

# Notice that \(bc - ad\) is the negative of \(ad - bc\). Since \(ad - bc = -8\), it follows that:
# \[
# bc - ad = -(-8) = 8.
# \]

# Therefore, the determinant of the matrix \(\begin{vmatrix} b & a \\ d & c \end{vmatrix}\) is \(\boxed{8}\)."""


# solution_string = r"""To find $\cos{C}$ given that $\cos{B} = \frac{3}{5}$ in triangle $ABC$, we can use trigonometric identities.

# Firstly recall that $\cos^2{\theta} + \sin^2{\theta} = 1$ for any angle $\theta$. We need to find $\sin{B}$ first:

# Given $\cos{B} = \frac{3}{5}$,

# $\sin^2{B} = 1 - \cos^2{B} = 1 - (\frac{3}{5})^2 = 1 - \frac{9}{25} = \frac{16}{25}$,

# So $\sin{B} = \sqrt{\frac{16}{25}} = \frac{4}{5}$.

# Now consider triangle ABC where angles sum up to $\pi$ radians ($180^\circ$):

# $\cos{C} = -\cos{(A+B)}$ because $\cos$ function is negative in the second quadrant where angle C lies.

# Using cosine addition formula $\cos(A+B) = \cos{A}\cos{B} - \sin{A}\sin{B}$,

# We know $\cos{B} = \frac{3}{5}$,

# And since $\sin{B} = \frac{4}{5}$,

# We need $\sin{A}$ which equals $\cos{B}$ due to complementary angles property ($\sin(\frac{\pi}{2} - \theta) = \cos{\theta}$),

# So $\sin{A} = \frac{3}{5}$,

# Then $\cos{C} = -(\cos{A}\cos{B} - \sin{A}\sin{B}) = -( \frac{3}{5}*\frac{3}{5} - \frac{3}{5}*\frac{4}{5}) = -( \frac{9}{25} - \frac{12}{25}) = -( \frac{-3}{25}) = \frac{3}{25}$.

# Therefore,

# {"answer": "3/25"}"""

# print(combined_function(latex_string)) # Should process the LaTeX string and extract the boxed answer
# print(combined_function(solution_string)) # Should extract the answer from the solution
