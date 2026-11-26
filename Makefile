.PHONY: up up-gpu down logs rebuild lint test

up:
	docker compose up -d --build

up-gpu:
	docker compose -f compose.yaml -f compose.gpu.yaml up -d --build

down:
	docker compose down

logs:
	docker compose logs -f asr-api

rebuild:
	docker compose build --no-cache && docker compose up -d

lint:
	ruff check app tests examples

test:
	pytest -q
