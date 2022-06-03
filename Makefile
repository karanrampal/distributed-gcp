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
	mypy src tests --disable-error-code import --disable-error-code attr-defined\
		--disable-error-code call-arg # missing library stubs

lint:
	pylint -j 6 src tests --ignore .ipynb_checkpoints\
		-d E1101,E0611,W0511,R0801,R0913,R0914 # Beam and fixme

clean:
	rm -r __pycache__ .coverage .mypy_cache .pytest_cache *.log .ipynb_checkpoints dist

all: install lint test

.PHONY: lint format clean all