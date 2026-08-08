.PHONY: help install test run-app run-api docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies in virtual environment"
	@echo "  make test        - Run test suite with pytest"
	@echo "  make run-app     - Launch Streamlit analytics dashboard"
	@echo "  make run-api     - Launch FastAPI microservice"
	@echo "  make docker-up   - Launch multi-container platform via Docker Compose"
	@echo "  make docker-down - Stop Docker Compose containers"
	@echo "  make clean       - Remove cached bytecode and temp artifacts"

install:
	pip install -r requirements.txt

test:
	pytest -v tests/ --durations=10

run-app:
	streamlit run app/streamlit_app.py

run-api:
	uvicorn api.main:app --port 8000 --reload

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
