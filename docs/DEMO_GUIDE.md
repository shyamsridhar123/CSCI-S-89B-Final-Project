# 🎬 Comprehensive Demo Guide

## Financial Document Intelligence Pipeline - CSCI S-89B Final Project

This guide provides step-by-step instructions for demonstrating the complete Financial Document Intelligence Pipeline, from running the training notebooks to using the interactive dashboard.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Part 1: Environment Setup](#part-1-environment-setup)
3. [Part 2: Running the Notebooks](#part-2-running-the-notebooks)
4. [Part 3: Launching the Dashboard](#part-3-launching-the-dashboard)
5. [Part 4: User Experience Walkthrough](#part-4-user-experience-walkthrough)
6. [Screen Recording Tips](#screen-recording-tips)
7. [Demo Script](#demo-script)

---

## Prerequisites

Before starting the demo, ensure you have:

- ✅ Python 3.9+ installed (tested with 3.11)
- ✅ Conda or venv for environment management
- ✅ 4GB+ RAM available
- ✅ Internet connection (for initial model downloads)
- ✅ Screen recording software (e.g., OBS Studio, Loom, or built-in OS recorder)

---

## Part 1: Environment Setup

### Step 1.1: Clone the Repository

```bash
git clone <repository-url>
cd CSCI-S-89B-Final-Project
```

📸 **Screenshot opportunity**: Terminal showing successful clone

### Step 1.2: Create and Activate Environment

```bash
# Using conda (recommended)
conda create -n nlp-pipeline python=3.11
conda activate nlp-pipeline

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 1.3: Install Dependencies

```bash
pip install -r requirements.txt
```

📸 **Screenshot opportunity**: Terminal showing successful package installation

### Step 1.4: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

📸 **Screenshot opportunity**: Terminal showing spaCy model download complete

---

## Part 2: Running the Notebooks

The notebooks must be executed in sequence. Each notebook builds upon the previous one's outputs.

### Notebook Overview

| Order | Notebook | Purpose | Time |
|-------|----------|---------|------|
| 0 | `00_setup_models.ipynb` | Pre-download models (optional) | ~2-5 min |
| 1 | `01_data_preparation.ipynb` | Download SEC data, create splits | ~5-10 min |
| 2 | `02_train_classifier.ipynb` | Train Keras classifier | ~2-5 min |
| 3 | `03_finbert_enhanced_detection.ipynb` | FinBERT demo (optional) | ~2-3 min |

### Step 2.1: Launch Jupyter Notebook

```bash
jupyter notebook notebooks/
```

📸 **Screenshot opportunity**: Jupyter file browser showing all notebooks

### Step 2.2: Run Notebook 00 (Optional) - Setup Models

1. Open `00_setup_models.ipynb`
2. Click **Cell → Run All** or use `Shift+Enter` to run cells sequentially
3. Wait for all model downloads to complete

📸 **Screenshot opportunity**: Notebook showing successful model downloads

### Step 2.3: Run Notebook 01 - Data Preparation

1. Open `01_data_preparation.ipynb`
2. Execute all cells in order
3. **Key outputs to highlight**:
   - Dataset download from HuggingFace
   - Data exploration and statistics
   - Train/validation/test split creation
   - Sample documents saved

📸 **Screenshot opportunities**:
- Dataset loading output
- Data distribution visualization
- Output files created in `data/processed/`

**Expected output files**:
```
data/processed/section/
├── train.csv
├── val.csv
└── test.csv
```

### Step 2.4: Run Notebook 02 - Train Classifier

1. Open `02_train_classifier.ipynb`
2. Execute all cells in order
3. **Key outputs to highlight**:
   - Model architecture summary
   - Training progress with accuracy/loss
   - Training history plots
   - Confusion matrix
   - Test set evaluation metrics

📸 **Screenshot opportunities**:
- Model architecture summary
- Training progress (epochs completing)
- Training history chart showing accuracy/loss curves
- Confusion matrix visualization
- Classification report with precision/recall/F1

**Expected output files**:
```
models/
├── classifier_model.keras
├── vectorizer.joblib
├── label_encoder.joblib
└── classes.joblib

docs/
├── training_history.png
└── confusion_matrix.png
```

### Step 2.5: Run Notebook 03 (Optional) - FinBERT Enhancement

1. Open `03_finbert_enhanced_detection.ipynb`
2. Execute all cells in order
3. **Key outputs to highlight**:
   - FinBERT model loading
   - Comparison of baseline vs. FinBERT-enhanced detection
   - Sentiment-based severity scoring

📸 **Screenshot opportunities**:
- FinBERT model initialization
- Risk detection comparison (baseline vs. enhanced)
- Sentiment analysis examples

---

## Part 3: Launching the Dashboard

### Step 3.1: Start the Application

```bash
python app.py
```

📸 **Screenshot opportunity**: Terminal showing initialization messages

**Expected console output**:
```
============================================================
Initializing Financial Document Intelligence Pipeline
============================================================

[1/7] Loading Document Classifier...
[2/7] Loading Sentiment Analyzer...
[3/7] Loading Entity Extractor...
[4/7] Loading Risk Detector...
[5/7] Loading Metrics Extractor...
[6/7] Loading Forward-Looking Detector...
[7/7] Loading Summarizer...

============================================================
Pipeline Ready!
============================================================

Running on local URL:  http://localhost:7860
```

### Step 3.2: Access the Dashboard

Open your browser and navigate to: **http://localhost:7860**

📸 **Screenshot opportunity**: Dashboard initial state in browser

---

## Part 4: User Experience Walkthrough

### Step 4.1: Dashboard Overview

Upon loading, you'll see:

1. **Header**: Title and project information
2. **Input Area**: Text box for pasting financial documents
3. **Analysis Options**: Checkboxes to toggle specific analyses
4. **Analyze Button**: Triggers the NLP pipeline
5. **Results Area**: Accordion sections for each analysis type
6. **Sample Documents**: Dropdown to load pre-configured examples

📸 **Screenshot opportunity**: Full dashboard view with all sections visible

### Step 4.2: Load a Sample Document

1. Scroll to "Load Sample 10-K Filing" section
2. Select a company from the dropdown (e.g., "🍎 Apple Inc. (FY2024)")
3. Click "📥 Load Sample"
4. Observe the text area populated with SEC filing content

📸 **Screenshot opportunity**: Dropdown selection and loaded sample text

### Step 4.3: Configure Analysis Options

All analysis options are enabled by default:

- ☑️ **Document Classification**: Identifies SEC 10-K section type
- ☑️ **Sentiment Analysis**: FinBERT-based financial sentiment
- ☑️ **Named Entity Recognition**: Extracts organizations, money, dates
- ☑️ **Risk Detection**: Identifies risk factors by category
- ☑️ **Metrics & Forward-Looking**: Extracts financial metrics and predictions
- ☑️ **Summarization**: Generates extractive summary

📸 **Screenshot opportunity**: Analysis options panel

### Step 4.4: Run Analysis

1. Click the **"🔍 Analyze"** button
2. Observe the progress indicator
3. Wait for all analyses to complete (~3-5 seconds)

📸 **Screenshot opportunity**: Progress bar during analysis

### Step 4.5: Review Executive Summary

The **Executive Summary Card** appears at the top showing:

- Document Type (classified section)
- Classification Confidence
- Overall Sentiment (positive/neutral/negative badge)
- Risk Level (Low/Medium/High with score)
- Key Entities detected
- Forward-Looking Statement count

📸 **Screenshot opportunity**: Executive Summary Card with results

### Step 4.6: Explore Classification & Sentiment

Expand the **"📊 Classification & Sentiment"** accordion:

- **Left**: Sentiment gauge visualization (0-100 scale)
- **Right**: Classification probability bar chart (top 10 sections)

📸 **Screenshot opportunity**: Side-by-side sentiment and classification charts

### Step 4.7: Review Entity Extraction

Expand the **"🏷️ Entities"** accordion:

- **Left**: Entity distribution bar chart by type
- **Right**: Entity table with text, type, and count

📸 **Screenshot opportunity**: Entity visualization and table

### Step 4.8: Examine Risk Analysis

Expand the **"⚠️ Risk Analysis"** accordion:

- **Risk Score Gauge**: Overall risk level (0-100)
- **Category Breakdown**: Risk mentions by category (regulatory, financial, etc.)
- **Detailed Metrics**: Severity breakdown, forward-looking statement counts

📸 **Screenshot opportunity**: Risk analysis visualizations

### Step 4.9: Read Summary

Expand the **"📝 Summary"** accordion:

- View the 5 most important sentences extracted from the document
- These represent key information based on TF-IDF scoring

📸 **Screenshot opportunity**: Summary text output

### Step 4.10: Export Report

Expand the **"📥 Export Report"** accordion:

1. Click the download link for the JSON report
2. Open the downloaded file to view structured analysis data

📸 **Screenshot opportunity**: Export section and downloaded JSON file

### Step 4.11: Highlighted Text View

Expand the **"📄 Document Text (Highlighted)"** accordion:

- View the document with color-coded annotations:
  - 🔵 Blue: Entities
  - 🟢 Green: Money values
  - 🔴 Red: Risk indicators

📸 **Screenshot opportunity**: Highlighted text with annotations

### Step 4.12: Clear and Try Another Document

1. Click **"🗑️ Clear"** to reset all fields
2. Either paste your own text or load another sample
3. Repeat the analysis process

📸 **Screenshot opportunity**: Cleared dashboard ready for new input

---

## Screen Recording Tips

### Recommended Recording Software

- **OBS Studio** (Free, cross-platform)
- **Loom** (Easy sharing, browser-based)
- **macOS**: Built-in QuickTime Player
- **Windows**: Built-in Xbox Game Bar (Win+G)

### Recording Settings

- **Resolution**: 1920x1080 (Full HD) or 1280x720 (HD)
- **Frame Rate**: 30 FPS
- **Audio**: Include microphone for narration

### Demo Flow for Recording

1. **Introduction** (~1 min)
   - Show repository structure
   - Explain project purpose

2. **Notebook Execution** (~5 min)
   - Show running Notebook 01 (data prep)
   - Show running Notebook 02 (training)
   - Highlight key outputs

3. **Dashboard Demo** (~5 min)
   - Launch app.py
   - Load sample document
   - Run full analysis
   - Walk through each result section

4. **Conclusion** (~1 min)
   - Summarize capabilities
   - Mention potential use cases

### Narration Tips

- Speak clearly and at a moderate pace
- Explain what you're doing before each action
- Pause briefly on important results
- Highlight the value of each NLP component

---

## Demo Script

### Introduction (30 seconds)

> "Welcome to the Financial Document Intelligence Pipeline demo. This project was developed for CSCI S-89B at Harvard Extension School. It demonstrates practical NLP applications for analyzing SEC filings and financial documents."

### Notebook Execution (2 minutes)

> "First, let's run the training notebooks. We'll start with data preparation to download SEC 10-K filings from HuggingFace..."

> "Now we'll train the document classifier. This Keras neural network will learn to identify 20 different SEC section types..."

> "As you can see, training achieves high accuracy on the validation set. The confusion matrix shows strong performance across all section types."

### Dashboard Launch (30 seconds)

> "With our models trained, let's launch the Gradio dashboard by running app.py..."

> "The pipeline loads seven NLP components: document classifier, sentiment analyzer, entity extractor, risk detector, metrics extractor, forward-looking detector, and summarizer."

### Analysis Demo (3 minutes)

> "Let's load Apple's 10-K filing as our sample document..."

> "I'll click Analyze to run the full NLP pipeline. Notice the progress bar as each component processes the text..."

> "The executive summary shows this is an Item 7 - Management Discussion & Analysis section, with neutral sentiment and medium risk level..."

> "Looking at the sentiment gauge, we can see the document is primarily neutral, which is typical for regulatory filings..."

> "The entity extraction identified key organizations, monetary values, and dates. Apple Inc. is the most frequently mentioned organization..."

> "Risk analysis detected mentions across multiple categories including regulatory, operational, and market risks..."

> "Finally, the extractive summary captures the most important sentences from the document..."

### Conclusion (30 seconds)

> "This pipeline demonstrates how modern NLP techniques can automate financial document analysis. It combines deep learning classification, transformer-based sentiment analysis, and traditional NLP methods for a comprehensive solution."

---

## 📸 Screenshot Checklist

Use this checklist to ensure you capture all necessary screenshots:

### Environment Setup
- [ ] Repository cloned successfully
- [ ] Dependencies installed
- [ ] spaCy model downloaded

### Notebooks
- [ ] Jupyter file browser with all notebooks
- [ ] Notebook 01: Dataset loading
- [ ] Notebook 01: Data distribution
- [ ] Notebook 02: Model architecture
- [ ] Notebook 02: Training progress
- [ ] Notebook 02: Training history chart
- [ ] Notebook 02: Confusion matrix
- [ ] Notebook 03: FinBERT comparison (optional)

### Dashboard
- [ ] Terminal showing app.py startup
- [ ] Dashboard initial state
- [ ] Sample document loaded
- [ ] Analysis in progress
- [ ] Executive Summary Card
- [ ] Sentiment & Classification charts
- [ ] Entity distribution and table
- [ ] Risk analysis visualizations
- [ ] Summary output
- [ ] Export/Download section
- [ ] Highlighted text view

---

## 🎥 Video Presentation Checklist

- [ ] Record screen and audio
- [ ] Cover all major sections
- [ ] Duration: 8-12 minutes recommended
- [ ] Export in MP4 format
- [ ] Upload to YouTube, Loom, or course platform
- [ ] Add link to README.md

---

## Troubleshooting During Demo

### "Classifier not trained" warning
**Solution**: Run Notebook 02 first to create model files in `models/` directory.

### Dashboard won't start
**Solution**: Ensure port 7860 is available, or specify a different port:
```bash
python app.py --port 7861
```

### Analysis takes too long
**Solution**: Disable some analysis options for faster demo, or use a shorter document.

### Models downloading during demo
**Solution**: Run Notebook 00 beforehand to pre-download all models.

---

## Additional Resources

- **README.md**: Complete project documentation
- **notebooks/README.md**: Detailed notebook guide
- **docs/enhancement_plan.md**: Future improvements roadmap

---

*This demo guide was created for CSCI S-89B Final Project - Financial Document Intelligence Pipeline*
