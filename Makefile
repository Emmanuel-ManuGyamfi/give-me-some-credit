.PHONY: setup data train evaluate submit lint

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	# Requires 'kaggle' CLI configured with API token
	mkdir -p data/raw
	kaggle competitions download -c GiveMeSomeCredit -p data/raw
	unzip -o data/raw/GiveMeSomeCredit.zip -d data/raw

train:
	python -m src.train

evaluate:
	python -m src.evaluate

submit:
	python -m src.make_submission

lint:
	python -m pip install ruff && ruff check src
