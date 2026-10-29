.PHONY: setup lint test run-api web docker-build docker-run train data evaluate drift

setup:
	pip install -r requirements-dev.txt
	pip install -e .
	npm --prefix frontend install

lint:
	ruff check .

test:
	pytest

run-api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

web:
	npm --prefix frontend run dev

data:
	python scripts/generate_demo_data.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

drift:
	python scripts/generate_drift_report.py

docker-build:
	docker build -t transactionguard .

docker-run:
	docker run -p 8000:8000 transactionguard
