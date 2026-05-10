.PHONY: install test lint format download detect track eval render-demo demo docker clean

PY ?= python
TRACKER ?= bytetrack

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

format:
	ruff format src tests
	ruff check --fix src tests

download:
	$(PY) scripts/download_kitti.py

detect:
	tracking detect --config configs/detector_yolov8.yaml

track:
	tracking track --config configs/tracker_$(TRACKER).yaml

eval:
	bash scripts/run_eval.sh

render-demo:
	$(PY) scripts/render_demo.py

demo:
	streamlit run streamlit_app/app.py

docker:
	docker build -t week1-tracking:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
