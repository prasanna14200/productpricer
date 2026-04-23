# Product Pricer

An end-to-end machine learning project that fine-tunes an open-source LLM to predict product prices from descriptions.

## Overview

This project demonstrates a complete ML pipeline:
- Data curation from Amazon reviews datasets
- Fine-tuning GPT-2 for price prediction
- Inference and deployment via Gradio web app

## Setup

1. Clone this repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` (Hugging Face token).
4. Run data curation: `python src/data_curation.py --categories "Home Appliances" --output data/processed/train.pkl`
5. Train the model: `python src/train.py --data data/processed/fine_tune_train.jsonl`
6. Run inference: `python src/inference.py --description "A red coffee maker"`
7. Launch web app: `python src/app.py`

## Usage

- Data curation: `python src/data_curation.py [options]`
- Training: `python src/train.py [options]`
- Inference: `python src/inference.py --description "Product description"`
- Web app: `python src/app.py`

## Demo

Run `python src/app.py` to launch a local Gradio interface.

For Hugging Face Spaces: Upload the `src/app.py` and model files to a new Space.

## Project Structure

- `data/`: Raw and processed datasets
- `src/`: Core scripts
- `models/`: Saved models
- `notebooks/`: Original exploration notebooks
- `tests/`: Unit tests