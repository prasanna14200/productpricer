# Verification Checklist

Run these commands to verify everything works:

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Run Tests
```bash
python -m pytest tests/ -v
```

## 3. Test Imports
```bash
python -c "from src.items import Item; print('✓ items.py')"
python -c "from src.loaders import ItemLoader; print('✓ loaders.py')"
python -c "from src.data_curation import main; print('✓ data_curation.py')"
python -c "from src.train import main; print('✓ train.py')"
python -c "from src.inference import predict_price; print('✓ inference.py')"
python -c "from src.app import iface; print('✓ app.py')"
```

## 4. Test Inference
```bash
python src/inference.py --description "A red coffee maker"
```

## 5. Test Web App
```bash
python src/app.py
```
Then open http://localhost:7860 in your browser.

## 6. Check Data Files
```bash
ls -la data/processed/
```

All files should be present and non-empty.

## 7. Check Model Files
```bash
ls -la models/gpt2-pricer/
```

Should contain config.json, pytorch_model.bin, tokenizer files, etc.