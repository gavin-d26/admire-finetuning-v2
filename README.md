# ADMIRE Vision-Language Model Finetuning

### Setup
- Install Pyenv: https://github.com/pyenv/pyenv
- Install the Python version specified in `.python-version` with Pyenv: `pyenv install 3.12.6`
- Create virtual environment: `python -m venv venv`
- Activate virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

If you choose to download additional packages (eg `seaborn`), do the following steps:
1. `pip install seaborn`
2. `pip freeze > requirements.txt`

If pre-commit hooks don't seem to be working, try the following:
1. `pre-commit install`
2. `pre-commit run --all-files`
This ensures that everyone has access the same pinned, resolved versions of dependencies.
