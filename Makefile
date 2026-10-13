VENV := .venv
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip

.PHONY: install run test clean

install:
	python -m venv $(VENV)
	$(PIP) install -e ".[dev]"

run:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache photoproof/__pycache__ tests/__pycache__
