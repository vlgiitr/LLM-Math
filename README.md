# 

# MATH Vs LLMs
This repo contains official code for the paper [Give me a hint: Can LLMs take a hint to solve math
 problems?](https://openreview.net/pdf?id=eeZG97VjYa)
If you find our code or paper useful, please consider citing
```
@inproceedings{agrawal2024give,
title={Give me a hint: Can {LLM}s take a hint to solve math problems?},
author={Vansh Agrawal and Pratham Singla and Amitoj Singh Miglani and Shivank Garg and Ayush Mangal},
booktitle={The 4th Workshop on Mathematical Reasoning and AI at NeurIPS'24},
year={2024},
url={https://openreview.net/forum?id=eeZG97VjYa}
}
```


## Project Overview
This project evaluates the mathematical capabilities of Large Language Models (LLMs) and examines how providing hints in prompts affects their performance. Further we look at how adverserial hinting and examples can misdirect the answers of the LLMs and look at capabilites an LLM to get out of that direction and realise the mistake.
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
cd Evaluation
python Evaluation/baseline.py
```

Similary, To generate other results:
```bash
cd Evaluation
python {filename}.py
```

## Dataset
We use the test_Math dataset for evaluation on all the models. It consists of problems from the following 7 categories:
- Algebra
- Counting and probability
- Geometry
- Intermediate algebra
- Number theory
- Prealgebra
- Precalculus

A sample of this dataset is provided in the repo under the test_Math folder with each sub-folder containing 2-3 problems of the given category 

