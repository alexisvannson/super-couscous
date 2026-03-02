VENV = .venv
PYTHON = $(VENV)/bin/python
.PHONY: install clean train-model test

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

%:
	@:
