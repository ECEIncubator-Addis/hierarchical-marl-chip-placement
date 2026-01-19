# Makefile for preprocessing TILOS MacroPlacement dataset
# Usage:
#   make run                  # Process default design (ariane136 on Nangate45)
#   make run DESIGN=MemPool_tile PDK=ASAP7   # Override design/PDK
#   make run-all              # Process multiple common designs (customize list below)
#   make clean                # Remove processed output directory
#   make clone                # Clone the repo if missing

# --------------------------------------------------------------------------
# Configuration (edit these as needed)
# --------------------------------------------------------------------------

# Default values
DESIGN ?= ariane136
PDK ?= Nangate45

# Directory containing this Makefile
THIS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Repository root (two levels up from this Makefile)
REPO_ROOT := $(abspath $(THIS_DIR)/../)

# Output directory for processed summaries/metadata
OUTPUT_DIR := $(REPO_ROOT)/processed

# Python script location
SCRIPT := $(REPO_ROOT)/load_datasets.py

# Designs
ALL_DESIGNS := ariane136 ariane133 MemPool_tile MemPool_group NVDLA BlackParrot

# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

.PHONY: all run run-all clone clean

# Default target
all: run

# Clone the repository if it doesn't exist
clone:
	@if [ ! -d "$(REPO_ROOT)" ]; then \
		echo "Cloning TILOS MacroPlacement repository..."; \
		git clone https://github.com/TILOS-AI-Institute/MacroPlacement.git $(REPO_ROOT); \
	else \
		echo "$(REPO_ROOT) already exists. Pull latest changes? Run 'git -C $(REPO_ROOT) pull' manually."; \
	fi

# Run preprocessing for a single design (override with make run DESIGN=... PDK=...)
run: clone
	@echo "Preprocessing design: $(DESIGN) with PDK: $(PDK)"
	python $(SCRIPT) --repo-root $(REPO_ROOT) --design $(DESIGN) --pdk $(PDK) --output-dir $(OUTPUT_DIR)

# Run preprocessing for all listed designs (uses default PDK; override PDK globally if needed)
run-all: clone
	@for design in $(ALL_DESIGNS); do \
		echo "Processing $$design with PDK $(PDK)"; \
		python $(SCRIPT) --repo-root $(REPO_ROOT) --design $$design --pdk $(PDK) --output-dir $(OUTPUT_DIR) || exit 1; \
	done
	@echo "All designs processed!"

# Clean processed outputs
clean:
	@echo "Removing $(OUTPUT_DIR)/ directory..."
	@rm -rf $(OUTPUT_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  make run          - Process single design (default: $(DESIGN) on $(PDK))"
	@echo "  make run DESIGN=X PDK=Y - Custom design/PDK"
	@echo "  make run-all      - Process all designs in list with default PDK"
	@echo "  make clone        - Clone the repo"
	@echo "  make clean        - Remove processed outputs"
	@echo "  make help         - Show this message"