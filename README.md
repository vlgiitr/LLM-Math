# MATH VS LLMs

This project evaluates the mathematical capabilities of Large Language Models (LLMs) and examines how providing hints in prompts affects their performance. We also look at the ability of VLMs to solve geometrical questions. Further we look at how adverserial hinting can misdirect the answers of the LLMs and look at capabilites an LLM to get out of that direction and realise the mistake.

## Project Overview
Here we prompt the mathematical problems to LLMs, with and without hints, and analyzes the responses to assess:

- The baseline mathematical ability of LLMs
- How hint-enhanced prompts impact the accuracy and approach of LLM solutions
- How adverserial hints affect the problem solving of an LLM
- Test the capabilites of VLMs on mathematical problems

# Installation
```bash
git clone https://github.com/vlgiitr/LLM-Math.git
pip install -r requirements.txt
```
To generate the baseline:
```bash
cd Evaluation/
python baseline.py
```

## Contributing
Contributions are welcome! Please read our Contributing Guide for more information.
## License
This project is licensed under the  - see the LICENSE file for details.