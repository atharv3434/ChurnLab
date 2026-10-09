export PYTHONPATH := src
train:
	python -m churnlab.train
test:
	pytest -q
serve:
	uvicorn churnlab.api:app --reload
