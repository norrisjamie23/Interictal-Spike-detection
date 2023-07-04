initialize_git:
	@echo "Initializing git..."
	git init 
	
install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install

activate:
	@echo "Activating virtual environment"
	poetry shell

download_data:
	@echo "Downloading data..."
	# wget https://gist.githubusercontent.com/khuyentran1401/a1abde0a7d27d31c7dd08f34a2c29d8f/raw/da2b0f2c9743e102b9dfa6cd75e94708d01640c9/Iris.csv -O data/raw/iris.csv

setup: initialize_git install download_data

test:
	pytest

docs_view:
	@echo View API documentation... 
	PYTHONPATH=src pdoc src --http localhost:8080

docs_save:
	@echo Save documentation to docs... 
	PYTHONPATH=src pdoc src -o docs

raw_data_path:=
raw_data_path_opt:=$(addprefix raw_data_path=,$(raw_data_path))

model_dir:=
model_dir_opt:=$(addprefix model_dir=,$(model_dir))

train:
	@echo "Preprocessing data and then training an NMF model on it"
	python src/train_workflow.py $(raw_data_path_opt)

detect:
	@echo "Preprocessing data and running detection on it"
	python src/detect_workflow.py $(raw_data_path_opt) $(model_dir_opt)

notebooks/results.ipynb: models/svc.pkl src/run_notebook.py
	@echo "Running notebook..."
	python src/run_notebook.py

pipeline: data/processed/xy.pkl models/svc.pkl notebooks/results.ipynb

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache