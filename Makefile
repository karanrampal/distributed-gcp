SHELL := /bin/bash
CONDAENV := environment.yaml

install: $(CONDAENV)
	conda env create -f $(CONDAENV)

install_ci: requirements.txt
	pip install --upgrade pip &&\
		pip install -r requirements.txt

build:
	python -m build

test:
	pytest -vv --cov --disable-warnings

format:
	black src tests
	isort src tests
	mypy src tests

lint:
	pylint -j 6 src tests

clean:
	rm -r .coverage .mypy_cache .pytest_cache .ipynb_checkpoints dist

all: install lint test

.PHONY: lint format clean all