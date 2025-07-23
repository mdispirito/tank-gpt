
.PHONY: help build up preprocess finetune clean

help:
	@echo "  make build                                    - Build Docker image only"
	@echo "  make up                                       - Build Docker image and start API server"
	@echo "  make preprocess [TARGET_SENDER=name]          - Run input data preprocessing"
	@echo "  make finetune                                 - Run PyTorch fine-tuning"
	@echo "  make clean                                    - Clean up generated output and model files"

build:
	docker build -t tank-gpt .

up:
	docker-compose up --build

preprocess:
	python src/preprocess.py --target-sender "$(TARGET_SENDER)"

finetune:
	python src/finetune.py

clean:
	rm -rf assets/output/*
	rm -rf assets/models/*
