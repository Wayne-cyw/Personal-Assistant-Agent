.PHONY: run test lint

run:
	uvicorn app.main:app --reload

test:
	pytest; status=$$?; [ $$status -eq 0 ] || [ $$status -eq 5 ]

lint:
	ruff check .
	mypy app
