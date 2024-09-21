# Makefile for TimeVQVAE project

# Default values
ADEP ?= EHAM
ADES ?= LIMC
DATA_SOURCE ?= EuroControl

# Directories
RAW_DATA_DIR = ../raw_data
DATA_DIR = data/real
SYNTHETIC_DATA_DIR = data/synthetic
SCRIPT_DIR = timevqvae/scripts
MODEL_DIR = saved_models


# Find the data file
DATA_FILE = $(shell find $(DATA_DIR) -type f -name "*$(DATA_SOURCE)*$(ADEP)*$(ADES)*.pkl" -print -quit)

# Check for trained models
MODEL_SUBDIR = $(MODEL_DIR)/$(DATA_SOURCE)_$(ADEP)_$(ADES)
TRAINED_MODELS = $(wildcard $(MODEL_SUBDIR)/stage*.ckpt)

# Check for generated synthetic data
GENERATED_FILE = $(shell find $(SYNTHETIC_DATA_DIR) -type f -name "*$(DATA_SOURCE)*$(ADEP)*$(ADES)*.pkl" -print -quit)

# Main target
all: generate

# Generate target
generate: check_models
	poetry run generate --config configs/config.yaml --dataset_file $(DATA_FILE) --model_save_dir $(MODEL_DIR) --synthetic_save_dir $(SYNTHETIC_DATA_DIR)

# Evaluate target
evaluate: check_models
	poetry run evaluate --config configs/config.yaml --dataset_file $(DATA_FILE) --model_save_dir $(MODEL_DIR)

# Evaluate flyability target
evaluate_flyability: check_generated
	poetry run evaluate_flyability --dataset_file $(DATA_FILE) --synthetic_data_file $(GENERATED_FILE)

# Check if synthetic data is generated, if not, run generate

check_generated:
ifeq ($(GENERATED_FILE),)
	@echo "Generated synthetic data file not found. Running generation..."
	$(MAKE) generate
else
	@echo "Using existing generated synthetic data file: $(GENERATED_FILE)"
endif


# Check if trained models exist, train if not
check_models:
ifeq ($(TRAINED_MODELS),)
	@echo "Trained models not found. Running training..."
	$(MAKE) train
else
	@echo "Using existing trained models: $(TRAINED_MODELS)"
endif

# Train target
train: check_data
	poetry run train --config configs/config.yaml --dataset_file $(DATA_FILE) --model_save_dir $(MODEL_DIR)

# Check if data file exists, preprocess if not
check_data:
ifeq ($(DATA_FILE),)
	@echo "Data file not found. Running preprocessing..."
	$(MAKE) preprocess
else
	@echo "Using existing data file: $(DATA_FILE)"
endif

# Preprocess data
preprocess:
	poetry run preprocess --ADEP $(ADEP) --ADES $(ADES) --data_source $(DATA_SOURCE) --raw_data_dir $(RAW_DATA_DIR) --save_dir $(DATA_DIR)

fix:
	black . && isort . 

app:
	streamlit run deployment/app.py

llm:
	python deployment/serve.py


# Phony targets
.PHONY: all generate check_models train check_data preprocess

# Help target
help:
	@echo "Usage:"
	@echo "  make [target] ADEP=<departure_airport> ADES=<arrival_airport> DATA_SOURCE=<EuroControl|OpenSky>"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)  - Check for models, data, and run generation"
	@echo "  generate       - Run generation (train if needed)"
	@echo "  evaluate       - Run evaluation (train if needed)"
	@echo "  evaluate_flyability - Run flyability evaluation (generate if needed)"
	@echo "  train          - Run training (preprocess if needed)"
	@echo "  preprocess     - Run preprocessing"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Example:"
	@echo "  make ADEP=EHAM ADES=LIMC DATA_SOURCE=EuroControl"