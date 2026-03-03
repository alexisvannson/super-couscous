VENV = .venv
PYTHON = $(VENV)/bin/python
.PHONY: install clean train-model train-densenet test evaluate cam help

help:
	@echo "Usage: make <target> [ARGS=...]"
	@echo ""
	@echo "Targets:"
	@echo "  install          Create virtualenv and install dependencies"
	@echo "  test             Run all tests with pytest"
	@echo "  train-model <m>  Train any model by name  (e.g. make train-model densenet)"
	@echo "  train-densenet   Train DenseNet-121 with configs/densenet.yaml(load pretrained weights)"
	@echo "  evaluate <m>     Evaluate model on test set with bootstrap F1 CIs"
	@echo "  cam <m>          Generate CAM heatmaps for an image"
	@echo "  clean            Remove venv, caches and build artefacts"

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf *.egg-info

test:
	$(PYTHON) -m pytest tests/ -v

train-model:
	$(PYTHON) scripts/training.py $(word 2, $(MAKECMDGOALS)) $(ARGS)

train-densenet:
	$(PYTHON) scripts/training.py densenet $(ARGS)

evaluate:
	$(PYTHON) scripts/evaluate.py $(word 2, $(MAKECMDGOALS)) $(ARGS)

cam:
	$(PYTHON) scripts/cam.py $(word 2, $(MAKECMDGOALS)) $(ARGS)

%:
	@:
