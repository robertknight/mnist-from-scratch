test:
	PYTHONPATH=. pytest .

format:
	black *.py
