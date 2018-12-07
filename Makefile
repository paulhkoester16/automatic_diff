.PHONY: test test-cover clean-pyc rm-cover-dir rm-coverage rm-cover lint install develop clean-dev build-dev

package=automatic_diff

clean-pyc:
	find . -name *__pycache__* -exec rm -rf {} +

rm-cover-dir:
	rm -rf cover

rm-coverage:
	rm .coverage

rm-cover: rm-cover-dir rm-coverage

clean-dev: clean-pyc
	python setup.py develop --uninstall

install: clean-pyc
	python setup.py install

develop: clean-dev
	python setup.py develop

test: develop
	nosetests

test-cover: develop rm-cover
	nosetests --with-coverage --cover-package=$(package) --cover-inclusive --cover-html

lint: clean-pyc
	pylint $(package)

build-dev: lint test-cover

