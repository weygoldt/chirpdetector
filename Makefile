.PHONY: docs install clean

# deploy docs to github pages
docs: 
	mkdocs gh-deploy --force

# install dependencies and package in editable mode
install:
	pip install --upgrade pip
	pip install setuptools wheel
	pip install -r requirements.txt
	pip install -e .

# clean up build files
clean:
	rm -rf site
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf .cache
	rm -rf .tox
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .vscode
	rm -rf .ipynb_checkpoints
	rm -rf .DS_Store

fmt:
	ruff check --fix-only ./chirpdetector ./tests
	ruff format ./chirpdetector ./tests

test:
	ruff check ./chirpdetector ./tests
	mypy ./chirpdetector
	pytest --cov --cov-report=xml

