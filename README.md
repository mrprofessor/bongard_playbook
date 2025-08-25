# Bongard Playbook

Evaluation playbook for the [Bongard-OpenWorld benchmark](https://github.com/rujiewu/Bongard-OpenWorld) - a benchmark that challenges models to identify concepts from positive/negative image sets using real-world images and open-vocabulary concepts.

## Dataset Preparation

Before running evaluations, you need to prepare the Bongard-OpenWorld dataset from the [original repository](https://github.com/rujiewu/Bongard-OpenWorld):

### Download Dataset

1. Download the backup dataset from the [Google Drive link](https://drive.google.com/drive/folders/1dS_FU8Gv7JcZJDVBhE-gQNKAyKpJ2qeJ) provided in the original repository
2. Extract the images to create this directory structure:
   ```
   assets/data/bongard-ow/
   ├── images/
   │   ├── 0000/
   │   ├── 0001/
   │   └── ... (up to 1009)
   ├── bbox_data.pkl
   ├── bongard_ow.json
   ├── bongard_ow_train.json
   ├── bongard_ow_val.json
   └── bongard_ow_test.json
   ```

### Important Notes

- Images must be extracted to `assets/data/bongard-ow/images/` with numbered folders
- Ensure you have the metadata files (`*.json` and `*.pkl`) in the root `bongard-ow/` directory
- The dataset contains images sourced from Google Images for research purposes

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.13+
- uv package manager

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd bongard_playbook
   ```

2. Install dependencies with uv:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

## Usage

The project includes various scripts for evaluating models on Bongard problems:

- `scripts/cvr.py` - Computer Vision Reasoning evaluation
- `scripts/cmr.py` - Comparative Multi-modal Reasoning evaluation  
- `scripts/sdr.py` - Structured Description Reasoning evaluation
- `scripts/captions.py` - Generate captions for images
- `scripts/scaptions.py` - Generate structured captions for images
- `scripts/eval.py` - Evaluation utilities

### Example Commands

```bash
# Comparative Multi-modal Reasoning with VLM and LLM
python scripts/cmr.py --vlm gpt5 --llm gpt5

# Structured Description Reasoning
python scripts/sdr.py --vlm gpt5 --llm gpt5

# Generate captions
python scripts/captions.py --model gpt5

# Computer Vision Reasoning
python scripts/cvr.py --model gpt5

# Generate structured captions
python scripts/scaptions.py --model gpt5
```

### Model Types

Models are categorized into three types in `scripts/constants.py`:

- **AIO_MODELS**: All-in-One models (multi-modal) - can accept both images and text as input
- **VLM_MODELS**: Vision-Language Models - specialized for image-text tasks  
- **LLM_MODELS**: Large Language Models - text-only models

**Note**: AIO models are multi-modal and can be used wherever VLMs or LLMs are required, but LLMs are text-only and cannot process images directly.

## Project Structure

```
├── assets/data/bongard-ow/     # Bongard-OW dataset
├── scripts/                    # Main evaluation scripts
├── notebooks/                  # Jupyter notebooks for analysis
├── results/                    # Evaluation results
├── captions/                   # Generated captions
└── pyproject.toml              # Project configuration
```
