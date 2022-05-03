SHELL := /bin/bash
CONDAENV := environment.yml

install: environment.yml
	conda env create -f $(CONDAENV)

install_ci: requirements.txt
	pip install --upgrade pip &&\
		pip install -r requirements.txt

build:
	python -m build

test:
	pytest -vv --cov

format:
	black src tests
	isort src tests
	mypy src tests

lint:
	pylint -j 6 src tests

clean:
	rm -rf __pycache__ .coverage .mypy_cache .pytest_cache *.log

all: install lint test

.PHONY: lint format clean all