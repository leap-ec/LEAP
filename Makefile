help:
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo \# LEAP Makefile
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo \#
	@echo "#	make venv			Create a virtual environment"
	@echo "#	source venv/bin/activate	Activate it"
	@echo \#
	@echo "#	make setup			Install LEAP & minimal dependencies"
	@echo "#	make doc			Build docs (in docs/build/html/)"
	@echo "#	make dist			Create package (i.e. for PyPI)"
	@echo \#
	@echo "#	make pep8			Check for PEP8 compliance"
	@echo "#	make test			Run fast and slow test suites"
	@echo "#	make test-fast			Run fast test suite"
	@echo "#	make test-slow			Run slow test suite"
	@echo "#	make kernel			Setup Jupyter for tests"
	@echo "#	make test-jupyter		Test the Jupyter examples"
	@echo \#
	@echo \#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
	@echo


venv:
	python3 -m venv --system-site-packages ./venv
	@echo
	@echo Built virtual environment in ./venv
	@echo Run \'source venv/bin/activate\' to activate it!

.PHONY: doc setup test test-fast test-slow kernel test-jupyter clean

docs-dependencies:
	pip install .[documentation]

doc: docs-dependencies
	pip install -r docs/requirements.txt
        # The apidoc call is long because we need to tell it to
        # use the venv's version of sphinx-build
	sphinx-apidoc -f -o docs/source/ leap_ec/ SPHINXBUILD='python $(shell which sphinx-build)'
	cd docs && make html

setup:
	pip install -e .


dist:
	pip install setuptools wheel
	python setup.py sdist bdist_wheel

test-dependencies:
	pip install .[development]

test: test-dependencies
	# Default options are configured in pytest.ini
	# Skip jupyter tests, because they only work if the kernel is configured manually
	python -m pytest -m "not jupyter"

test-fast:
	python -m pytest -m "not slow and not jupyter and not stochastic"

test-slow:
	python -m pytest -m slow

kernel:
	# Setup a kernel for Jupyter with the name test-jupyter uses to find it
	python -m ipykernel install --user --name="LEAP_venv"

test-jupyter:
	# Won't work unless you have a 'LEAP_venv' kernel
	python -m pytest -m jupyter

pep8:
	# Check for PEP8 compliance in source directories
	flake8 leap examples

clean:
	cd docs && make clean
	find . -name *.pyc -delete
	find . -name __pycache__ -type d -exec rm -rf {} +
	find . -name *.egg-info -type d -exec rm -rf {} +

