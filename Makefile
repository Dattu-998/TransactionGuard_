.PHONY: setup lint test run-api docker-build docker-run train evaluate demo drift

setup:
	pip install -r requirements.txt
	pip install -e .[dev]

lint:
	ruff check .

test:
	pytest

run-api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t transactionguard .

docker-run:
	docker run -p 8000:8000 transactionguard

demo:
	python scripts/generate_demo_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

drift:
	python scripts/generate_drift_report.py
