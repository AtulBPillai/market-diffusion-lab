.PHONY: install run test docker-run

install:
	python -m pip install -r requirements.txt

run:
	python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	python -m pytest -q

docker-run:
	docker compose up --build

