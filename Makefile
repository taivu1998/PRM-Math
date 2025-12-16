.PHONY: install clean train inference evaluate

# Default config and checkpoint paths
CONFIG ?= configs/default.yaml
CHECKPOINT ?= checkpoints/merged_model
DATASET ?= gsm8k
N_CANDIDATES ?= 16
N_PROBLEMS ?= 100

# Installation
install:
	pip install --upgrade pip
	pip install -e .

# Cleaning
clean:
	rm -rf __pycache__ .pytest_cache
	find . -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Deep clean (removes checkpoints and logs too)
clean-all: clean
	rm -rf logs/* checkpoints/*

# Training wrapper
# Usage: make train CONFIG=configs/default.yaml
train:
	python scripts/train.py --config $(CONFIG)

# Inference wrapper
# Usage: make inference CHECKPOINT=checkpoints/merged_model
# Usage: make inference CHECKPOINT=checkpoints/merged_model PROBLEM="Solve: 2x + 5 = 15"
inference:
ifdef PROBLEM
	python scripts/inference.py --model_path $(CHECKPOINT) --problem "$(PROBLEM)"
else
	python scripts/inference.py --model_path $(CHECKPOINT)
endif

# Evaluation
# Usage: make evaluate CHECKPOINT=checkpoints/merged_model DATASET=gsm8k N_CANDIDATES=16
evaluate:
	python scripts/evaluate.py --model_path $(CHECKPOINT) --dataset $(DATASET) \
		--n_candidates $(N_CANDIDATES) --n_problems $(N_PROBLEMS)