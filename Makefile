VENV = .venv
PYTHON = $(VENV)/bin/python
.PHONY: install clean train-model

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

clean: ## Supprime les fichiers temporaires et le cache (__pycache__)
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf *.egg-info

train-model:
	$(PYTHON) scripts/training.py $(word 2, $(MAKECMDGOALS)) $(ARGS)

%:
	@:
