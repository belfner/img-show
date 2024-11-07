.PHONY: install uninstall build upload-test upload

clean:
	python setup.py clean --all

build:
	python setup.py sdist bdist_wheel

upload-test:
	python -m twine upload --repository testpypi dist/* --verbose

upload:
	python -m twine upload dist/*