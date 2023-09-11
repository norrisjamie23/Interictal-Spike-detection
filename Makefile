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

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache