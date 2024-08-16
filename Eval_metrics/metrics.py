#normalisation and parsing for absolute scores
def norm_parse(latex_strings):
    import re
    from sympy import simplify

    def extract_boxed_answer(latex_string):
        """
        Extracts the content inside the \boxed{} command from a LaTeX string.
    
        Parameters:
        latex_string (str): The LaTeX string containing the \boxed{} command.
    
        Returns:
        str: The content inside the \boxed{} command.
        """
        match = re.search(r'\\boxed{([^}]*)}', latex_string)
        if match:
            return match.group(1)
        return None

    def normalize_answer(answer):
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

# Example LaTeX strings with \boxed{} answers
    # latex_strings = [
    #     r"The answer is \boxed{\frac{2}{3}}.",
    #     r"Final result: \boxed{0.6667}.",
    #     r"Solution: \boxed{2/3}."
    # ]

# Extract and normalize answers
    for latex in latex_strings:
        boxed_answer = extract_boxed_answer(latex)
        if boxed_answer:
            normalized_answer = normalize_answer(boxed_answer)
            print(f"Original: {boxed_answer}, Normalized: {normalized_answer}")
            return normalized_answer
        else:
            print("No boxed answer found.")

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