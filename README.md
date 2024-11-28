[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/GhX0HoeG)

# Requirements
- Python 3.10

# Run the Code
## Create the environment 
```bash
python3.10 -m venv .env
source .env/bin/activate
python3.10 -m pip install -r requirements.txt
```


```bash
python3.10 -m src.main
```

## Code

Your code should be clean, well-written and documented. You should use Python 3.

### Structure

The code repository comes with a pre-defined structure to ease your efforts. It is highly advised you stick to it. The defined structure works best if you make use of OOP in Python. This is however not required.

- 📁 **`src/`** : Your code for experiments should be here
  - 🐍 `__init.py__` : Marks directory as Python package. DO NOT TOUCH!
  - 🐍 `agent.py` : Your implementation of agents.
  - 🐍 `bandit.py` : Your implementation of bandits.
  - 🐍 `main.py` : The starting point of your program.
- 📁 **`data/`** : All of your results should be here: `.csv` files, figures, etc.
- 📁 **`analysis/`** : _(optional)_ Scripts and notebooks you use to analyze the results.
- 🐍 `setup.py` : Sets up the project. DO NOT TOUCH!
- 📄 `requirements.txt` : List of all the required libraries used in your project.
- 📄 `.gitignore` : Files to be ignored by `git`. 

Let's dive into a bit more detail.

#### `src/`

Your implementation should be here (algorithms, agents, neural networks etc.). The `main.py` is a starting point of your program and it should generate all the (raw) results of your experiments. Remember, that these results should be stored in the **`data/`** directory.

#### `data/`

All your results should be here. This includes all the figures you use in your report as well as the files containing the raw data, for example `.csv`.

#### `analysis/`

If you decide to perform some additional analysis of your results or generate figures outside of your `main.py` script, put the code you used here. This directory should preferebly contain Jupyter Notebooks (`.ipynb` files) that load raw results from `data/` directory, analyze them and generate appropriate figures.

### Replication

You need to make sure the results are replicable. Achieving this is a bit different depending on the tools you use. Make sure early on that this step works well on your machine. Otherwise, we might not be able to verify that your code runs correctly.

#### `pip`

Follow these steps if you are using the `pip` package manager.

1. Download the [pipreqs](https://pypi.org/project/pipreqs/) package, using pip: `pip3 install pipreqs`
2. Generate the list of the libraries you use in your implementation: `pipreqs --force --savepath requirements.txt src/`
3. Verify that your project works:
   1. Run the setup script: `pip3 install .`
   2. Run your implementation: `python3 src/main.py`
   3. Your program should run without any problem and put the generated (raw) results in the `data/` directory.


#### Sample

After following the above steps, your `requirements.txt` file could look somewhat like this:

```
matplotlib==3.6.2
numpy==1.23.5
pandas==1.5.1
```

## Code Requirements

### 1. Reproducibility
- Use a fixed random seed (e.g., for NumPy and PyTorch) to ensure reproducibility.

### 2. Code Style
- Follow PEP8 guidelines.
- Use clear, descriptive variable and method names (avoid single-letter names).
- Avoid global variables.
- Remove unused code and minimize code duplication.

### 3. Documentation
- Document every non-trivial method or function with clear docstrings.
- Type hinting for methods and functions is recommended but not mandatory.
- Provide a README with:
  - Clear instructions on how to run the code.
  - Steps to reproduce experiments (including details on dependencies and fixed seeds).

### 4. Version Control
- Create a feature-specific branch from `main` (e.g., a `dev` branch) for new features or changes.
- Submit pull requests with a clear description of changes for code review by the TA.



## Plagiarism

We know that there are a lot of tutorials on Reinforcement Learning online. If you find them useful, please learn from them, but keep in mind that some of the most popular tutorials have mistakes in them 😲. Further, if we suspect that a group plagiarised and copied someone else's approach without proper understanding, we will schedule a meeting with them to verify their integrity or immediately fail them if the plagiarism is obvious.

## Tips and Resources

1. Type hinting is not required, but it can help your partner understand your code - https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
2. Git workshop by Cover - https://studysupport.svcover.nl/?a=1
3. YouTube Git tutorial - https://www.youtube.com/watch?v=RGOj5yH7evk
4. OOP in Python - https://www.youtube.com/watch?v=JeznW_7DlB0
5. How to document Python? - https://www.datacamp.com/tutorial/docstrings-python4

