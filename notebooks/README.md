# Notebooks Guide

This directory contains Jupyter notebooks for data preparation, model training, and experimentation.

## 📋 Execution Sequence

Run the notebooks in the following order:

### Step 0: Setup Models (Optional - First Time Only)
```
00_setup_models.ipynb
```
- **Purpose**: Downloads and caches all required models (FinBERT, spaCy, NLTK)
- **Output**: Models cached in HuggingFace/spaCy directories
- **Time**: ~2-5 minutes (depends on download speed)
- **Note**: Models also download automatically on first use, so this step is optional

### Step 1: Data Preparation
```
01_data_preparation.ipynb
```
- **Purpose**: Downloads SEC 10-K filings from HuggingFace and prepares training data
- **Input**: `JanosAudran/financial-reports-sec` dataset (downloaded automatically)
- **Output**: 
  - `data/processed/section/` - Train/val/test splits for section classification
  - `data/processed/sentiment/` - Train/val/test splits for sentiment classification
  - `data/sample_docs/` - Sample documents for testing
- **Time**: ~5-10 minutes (depends on download speed)
- **Requirements**: Internet connection, ~500MB disk space

### Step 2: Model Training
```
02_train_classifier.ipynb
```
- **Purpose**: Trains the Keras neural network for SEC section classification
- **Input**: `data/processed/section/` (from Step 1)
- **Output**:
  - `models/classifier_model.keras` - Trained Keras model
  - `models/vectorizer.joblib` - TF-IDF vectorizer
  - `models/label_encoder.joblib` - Label encoder
  - `models/classes.joblib` - Class definitions
  - `docs/training_history.png` - Training curves
  - `docs/confusion_matrix.png` - Evaluation results
- **Time**: ~2-5 minutes (GPU recommended)
- **Requirements**: TensorFlow/Keras, GPU optional

### Step 3: FinBERT Enhancement (Optional)
```
03_finbert_enhanced_detection.ipynb
```
- **Purpose**: Demonstrates FinBERT-enhanced risk detection and forward-looking statement analysis
- **Input**: Sample text (included in notebook)
- **Output**: Comparison of baseline vs FinBERT-enhanced detection
- **Time**: ~2-3 minutes
- **Requirements**: PyTorch, transformers, ~1GB GPU VRAM (or CPU)

---

## 🚀 Quick Start

```bash
# Activate your environment
conda activate nlp-pipeline  # or your environment name

# Run notebooks in sequence
jupyter notebook 01_data_preparation.ipynb
# Execute all cells, then:
jupyter notebook 02_train_classifier.ipynb
# Execute all cells, then:
jupyter notebook 03_finbert_enhanced_detection.ipynb  # Optional
```

---

## ⚠️ Important Notes

1. **Run in Order**: Notebooks must be run sequentially. Step 2 requires outputs from Step 1.

2. **GPU Recommended**: While all notebooks work on CPU, GPU significantly speeds up:
   - Model training (Step 2)
   - FinBERT inference (Step 3)

3. **Memory Requirements**:
   - Step 1: ~4GB RAM
   - Step 2: ~4GB RAM + 2GB GPU (if available)
   - Step 3: ~4GB RAM + 1GB GPU (if available)

4. **First Run**: FinBERT model (~500MB) downloads automatically on first use in Step 3.

5. **Reproducibility**: All notebooks use `random_state=42` for reproducible results.

---

## 📊 What Each Notebook Produces

| Notebook | Key Output | Used By |
|----------|------------|---------|
| 00_setup_models | Cached FinBERT, spaCy, NLTK | All notebooks |
| 01_data_preparation | Training data CSVs | 02_train_classifier |
| 02_train_classifier | Keras model + vectorizer | app.py (dashboard) |
| 03_finbert_enhanced | Demo only | Reference implementation |

---

## 🔧 Troubleshooting

### "No module named 'src'"
Run from the `notebooks/` directory, or add project root to path:
```python
import sys
sys.path.insert(0, '..')
```

### TensorFlow GPU Errors
Force CPU mode:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### FinBERT Loading Issues
Use PyTorch framework explicitly:
```python
pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")
```
