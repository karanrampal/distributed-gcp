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
	mypy src tests --disable-error-code import # Apache beam import error

lint:
	pylint -j 6 src tests --ignore .ipynb_checkpoints -d E0611,W0511 # Apache beam and fixme

clean:
	rm -rf __pycache__ .coverage .mypy_cache .pytest_cache *.log

all: install lint test

.PHONY: lint format clean all