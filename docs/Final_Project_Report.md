# Financial Document Intelligence Pipeline

**CSCI S-89B Introduction to Natural Language Processing**  
**Harvard Extension School**  
**Final Project Report**

---

**Author:** Shyam Sridhar  
**Date:** December 2024  
**Project Repository:** https://github.com/shyamsridhar123/CSCI-S-89B-Final-Project

---

## Abstract

This project presents a comprehensive Natural Language Processing (NLP) pipeline for analyzing SEC filings and financial documents. The system integrates seven distinct NLP components: (1) a custom Keras neural network for document classification across 20 SEC 10-K section types, (2) financial sentiment analysis using FinBERT, (3) named entity recognition combining spaCy with custom regex patterns, (4) risk factor detection with severity scoring across seven risk categories, (5) financial metrics extraction, (6) forward-looking statement detection, and (7) TF-IDF-based extractive summarization.

The pipeline is deployed through an interactive Gradio dashboard that provides real-time analysis with visualizations including sentiment gauges, entity distribution charts, risk score indicators, and downloadable JSON reports. The document classifier achieves high accuracy on SEC 10-K sections using a feedforward neural network with TF-IDF vectorization (3,000 features with unigrams and bigrams). FinBERT provides domain-specific sentiment analysis trained on financial text, offering significant improvements over general-purpose sentiment models.

The project demonstrates practical applications of NLP in the financial domain, addressing the challenge of extracting actionable insights from complex regulatory filings. Processing time averages 3-5 seconds per document on CPU hardware, making the system practical for real-world use by investors, analysts, and compliance professionals. All components are modular, well-documented, and reproducible through the provided Jupyter notebooks and source code.

---

**Page 1**

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [System Architecture](#3-system-architecture)
4. [Installation and Configuration](#4-installation-and-configuration)
5. [Component Implementation Details](#5-component-implementation-details)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Results and Visualizations](#7-results-and-visualizations)
8. [What Worked and What Did Not](#8-what-worked-and-what-did-not)
9. [Lessons Learned](#9-lessons-learned)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)
12. [Appendix: Complete Code Listings](#12-appendix-complete-code-listings)

---

**Page 2**

---

## 1. Problem Statement

### 1.1 Background

Financial analysts and investors face significant challenges when analyzing SEC filings:

1. **Volume**: Thousands of 10-K, 10-Q, and 8-K filings are submitted to the SEC annually. Each 10-K filing can span 100+ pages of dense text.

2. **Complexity**: Documents contain dense legal and financial language with domain-specific terminology that general NLP tools struggle to interpret correctly.

3. **Time**: Manual analysis of a single 10-K filing can take several hours, making it impractical to analyze multiple companies efficiently.

4. **Consistency**: Human analysis varies in quality, focus, and interpretation, leading to inconsistent insights across different analysts.

### 1.2 Project Objective

This project aims to develop an automated NLP pipeline that can:

- **Classify** document sections to identify the type of content (Business Overview, Risk Factors, Management Discussion & Analysis, etc.)
- **Analyze sentiment** using models specifically trained on financial text
- **Extract entities** including organizations, monetary values, percentages, and fiscal dates
- **Detect risk factors** and categorize them by type and severity
- **Extract financial metrics** such as revenue, EPS, margins, and guidance
- **Identify forward-looking statements** that indicate management's expectations
- **Generate summaries** that capture the key points of lengthy documents

### 1.3 Target Users

- Financial analysts conducting company research
- Investors performing due diligence
- Compliance professionals reviewing regulatory filings
- Academic researchers studying financial disclosures

---

**Page 3**

---

## 2. Dataset

### 2.1 Data Source

**Primary Dataset**: [JanosAudran/financial-reports-sec](https://huggingface.co/datasets/JanosAudran/financial-reports-sec) from Hugging Face

This dataset contains SEC 10-K filings parsed by section, providing labeled data for training the document classifier.

**Available Configurations**:
- `large_full` - Complete SEC filings with full text (used for training)
- `large_lite` - Lighter version with metadata
- `small_full` - Smaller subset (used for development/testing)

### 2.2 Dataset Statistics

| Split | Documents | Purpose |
|-------|-----------|---------|
| Training | ~500 | Model training |
| Validation | ~100 | Hyperparameter tuning |
| Test | ~100 | Final evaluation |
| **Total** | **~700** | |

### 2.3 Section Classes (20 Types)

The classifier is trained to recognize 20 SEC 10-K section types:

| Section Code | Human-Readable Label |
|--------------|---------------------|
| `section_1` | Item 1 - Business Overview |
| `section_1A` | Item 1A - Risk Factors |
| `section_1B` | Item 1B - Unresolved Staff Comments |
| `section_2` | Item 2 - Properties |
| `section_3` | Item 3 - Legal Proceedings |
| `section_4` | Item 4 - Mine Safety Disclosures |
| `section_5` | Item 5 - Market Information |
| `section_6` | Item 6 - Selected Financial Data |
| `section_7` | Item 7 - Management Discussion & Analysis |
| `section_7A` | Item 7A - Market Risk Disclosures |
| `section_8` | Item 8 - Financial Statements |
| `section_9` | Item 9 - Auditor Changes |
| `section_9A` | Item 9A - Controls and Procedures |
| `section_9B` | Item 9B - Other Information |
| `section_10` | Item 10 - Directors & Officers |
| `section_11` | Item 11 - Executive Compensation |
| `section_12` | Item 12 - Security Ownership |
| `section_13` | Item 13 - Related Transactions |
| `section_14` | Item 14 - Accountant Fees |
| `section_15` | Item 15 - Exhibits & Schedules |

### 2.4 Sample Documents

The repository includes sample 10-K excerpts from major companies for testing:
- Apple Inc. (FY2024)
- Microsoft Corporation (FY2024)
- Amazon.com, Inc. (FY2023)
- Alphabet Inc. (FY2023)
- Meta Platforms (10-K)

---

**Page 4**

---

## 3. System Architecture

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Financial Document                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: Document Classification                   │
│                                                                 │
│  • Text preprocessing (cleaning, normalization)                 │
│  • TF-IDF Vectorization (3000 features, unigrams + bigrams)     │
│  • Keras Neural Network (256→128→64→20)                         │
│                                                                 │
│  Output: Section Type + Confidence Score                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 2: Sentiment Analysis                        │
│                                                                 │
│  • FinBERT (ProsusAI/finbert)                                   │
│  • Financial-specific sentiment classification                  │
│                                                                 │
│  Output: Positive/Negative/Neutral + Confidence                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: Named Entity Recognition                  │
│                                                                 │
│  • spaCy (en_core_web_sm) for ORG, PERSON, GPE, DATE, MONEY     │
│  • Custom regex for PERCENTAGE, FISCAL_DATE, TICKER             │
│                                                                 │
│  Output: List of entities with types and positions              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 4: Risk Factor Detection                     │
│                                                                 │
│  • 70+ regex patterns across 7 risk categories                  │
│  • Severity assessment (high/medium/low)                        │
│  • Optional FinBERT-enhanced severity scoring                   │
│                                                                 │
│  Output: Risk mentions with categories and severity             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 5: Metrics & Forward-Looking                 │
│                                                                 │
│  • 14 financial metric patterns (revenue, EPS, margins, etc.)   │
│  • Forward-looking statement detection with confidence levels   │
│                                                                 │
│  Output: Extracted metrics + Forward-looking statements         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 6: Extractive Summarization                  │
│                                                                 │
│  • NLTK sentence tokenization                                   │
│  • TF-IDF sentence scoring with position weighting              │
│  • Top-N sentence extraction                                    │
│                                                                 │
│  Output: Summary (top 5 sentences)                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRADIO DASHBOARD                             │
│                                                                 │
│  • Executive Summary Card                                       │
│  • Sentiment Gauge + Classification Chart                       │
│  • Entity Distribution + Entity Table                           │
│  • Risk Score Gauge + Risk Category Breakdown                   │
│  • Summary Display + JSON Report Download                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**Page 5**

---

### 3.2 Project Structure

```
NLPFinalProject/
├── app.py                        # Main Gradio dashboard application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
│
├── src/                          # Source modules
│   ├── __init__.py
│   ├── document_classifier.py    # Keras neural network classifier
│   ├── sentiment_analyzer.py     # FinBERT sentiment analysis
│   ├── entity_extractor.py       # spaCy NER + custom patterns
│   ├── risk_detector.py          # Risk factor detection
│   ├── metrics_extractor.py      # Financial metrics + forward-looking
│   └── summarizer.py             # Extractive summarization
│
├── notebooks/                    # Jupyter notebooks
│   ├── README.md                 # Notebook execution guide
│   ├── 00_setup_models.ipynb     # Pre-download models (optional)
│   ├── 01_data_preparation.ipynb # Data download and preprocessing
│   ├── 02_train_classifier.ipynb # Model training and evaluation
│   └── 03_finbert_enhanced_detection.ipynb  # FinBERT hybrid approach
│
├── models/                       # Saved model artifacts
│   ├── classifier_model.keras    # Trained Keras model
│   ├── vectorizer.joblib         # TF-IDF vectorizer
│   ├── label_encoder.joblib      # Label encoder
│   └── classes.joblib            # Class list
│
├── data/
│   ├── processed/
│   │   ├── section/              # Section classification data
│   │   │   ├── train.csv
│   │   │   ├── val.csv
│   │   │   └── test.csv
│   │   └── sentiment/            # Sentiment analysis data
│   ├── raw/                      # Raw downloaded data
│   └── sample_docs/              # Sample SEC filings for testing
│       ├── apple_10k_2024.txt
│       ├── microsoft_10k_2024.txt
│       ├── amazon_10k_2023.txt
│       ├── alphabet_10k_2023.txt
│       └── meta_10k.txt
│
└── docs/                         # Documentation
    ├── description.md            # Project requirements
    ├── project_plan_v2.md        # Project plan
    └── Final_Project_Report.md   # This report
```

---

**Page 6**

---

## 4. Installation and Configuration

### 4.1 Prerequisites

- **Python**: 3.9 or higher (tested with Python 3.11)
- **RAM**: 4GB minimum (8GB recommended for FinBERT)
- **Disk Space**: ~3GB for models and dependencies
- **GPU**: Optional (CPU works fine; GPU accelerates FinBERT)
- **Operating System**: Linux, macOS, or Windows

### 4.2 Step-by-Step Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/shyamsridhar123/CSCI-S-89B-Final-Project.git
cd CSCI-S-89B-Final-Project
```

#### Step 2: Create a Virtual Environment

**Option A: Using Conda (Recommended)**
```bash
conda create -n nlp-pipeline python=3.11
conda activate nlp-pipeline
```

**Option B: Using venv**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate   # Windows
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
# Core ML/DL
tensorflow>=2.10.0
torch>=2.0.0
transformers>=4.30.0
accelerate>=0.20.0

# NLP
spacy>=3.5.0
nltk>=3.8

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
datasets>=2.14.0

# Visualization & Dashboard
gradio>=6.0.0
plotly>=5.14.0

# Utilities
joblib>=1.3.0
tqdm>=4.65.0

# Development
jupyter>=1.0.0
ipykernel>=6.25.0
```

#### Step 4: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

#### Step 5: Verify Installation

```bash
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import spacy; print(f'spaCy: {spacy.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
```

---

**Page 7**

---

### 4.3 Training the Classifier (Required Before Dashboard)

The document classifier must be trained before using the dashboard. Follow these steps:

#### Step 1: Prepare the Data

Open and run all cells in `notebooks/01_data_preparation.ipynb`:

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

This notebook:
- Downloads the SEC dataset from Hugging Face
- Preprocesses and cleans the text data
- Creates train/validation/test splits
- Saves processed data to `data/processed/section/`

**Expected Output Files:**
- `data/processed/section/train.csv`
- `data/processed/section/val.csv`
- `data/processed/section/test.csv`

#### Step 2: Train the Classifier

Open and run all cells in `notebooks/02_train_classifier.ipynb`:

```bash
jupyter notebook notebooks/02_train_classifier.ipynb
```

This notebook:
- Loads the processed training data
- Builds and trains the Keras neural network
- Evaluates model performance on the test set
- Saves the trained model and artifacts

**Expected Output Files:**
- `models/classifier_model.keras` — Trained Keras model
- `models/vectorizer.joblib` — TF-IDF vectorizer
- `models/label_encoder.joblib` — Label encoder for class mapping
- `models/classes.joblib` — List of class names

#### Step 3: (Optional) Explore FinBERT-Enhanced Detection

```bash
jupyter notebook notebooks/03_finbert_enhanced_detection.ipynb
```

This notebook demonstrates how FinBERT can enhance risk detection and forward-looking statement analysis by adding sentiment-based severity scoring.

### 4.4 Troubleshooting Common Issues

#### Issue 1: CuDNN Version Mismatch

**Error Message:**
```
Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0
```

**Solution:**
```bash
# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python app.py

# Or update CuDNN
conda install cudnn=9.3
```

#### Issue 2: TensorFlow Import Hangs

**Solution:** Add to the first cell of any notebook:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
```

#### Issue 3: spaCy Model Not Found

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

#### Issue 4: FinBERT Download Issues

FinBERT (~500MB) downloads automatically on first use. Ensure stable internet connectivity.

---

**Page 8**

---

## 5. Component Implementation Details

### 5.1 Document Classifier (Keras Neural Network)

**File:** `src/document_classifier.py`

#### Architecture

The classifier uses a feedforward neural network with the following architecture:

```
Input: TF-IDF Vector (3000 features, unigrams + bigrams)
    ↓
Dense(256, ReLU) + BatchNormalization + Dropout(0.4)
    ↓
Dense(128, ReLU) + BatchNormalization + Dropout(0.3)
    ↓
Dense(64, ReLU) + Dropout(0.2)
    ↓
Dense(20, Softmax) → Section Classification
```

#### Key Implementation Details

**TF-IDF Vectorization:**
```python
self.vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words='english',
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,            # Minimum document frequency
    max_df=0.95          # Maximum document frequency
)
```

**Model Definition:**
```python
model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(len(self.classes), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Training Configuration:**
- Optimizer: Adam (learning rate = 0.001)
- Loss: Sparse Categorical Crossentropy
- Early Stopping: Patience = 3, restore best weights
- Learning Rate Reduction: Factor = 0.5, Patience = 2

#### Usage Example

```python
from src.document_classifier import DocumentClassifier

# Load trained classifier
classifier = DocumentClassifier()
classifier.load("models")

# Classify a document
text = "This section describes our business operations..."
section_code, readable_label, confidence = classifier.predict_with_label(text)
print(f"Section: {readable_label} ({confidence:.1%})")

# Get all class probabilities
probs = classifier.predict_proba(text)
```

---

**Page 9**

---

### 5.2 Sentiment Analyzer (FinBERT)

**File:** `src/sentiment_analyzer.py`

#### Model

Uses [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on financial text for sentiment classification.

**Why FinBERT?**
- Trained specifically on financial news and documents
- Understands domain-specific terminology (e.g., "bearish," "bullish," "revenue miss")
- Significantly outperforms general-purpose sentiment models on financial text

#### Output

- **Labels:** positive, negative, neutral
- **Score:** Confidence (0.0 to 1.0)
- **Interpretation:** Human-readable description

#### Usage Example

```python
from src.sentiment_analyzer import SentimentAnalyzer

sentiment = SentimentAnalyzer()
result = sentiment.analyze("Revenue increased 15% year-over-year, beating analyst expectations.")

print(f"Sentiment: {result['label']}")        # positive
print(f"Confidence: {result['score']:.1%}")   # 92.3%
print(f"Interpretation: {result['interpretation']}")
```

### 5.3 Entity Extractor (spaCy + Custom Patterns)

**File:** `src/entity_extractor.py`

#### Two-Stage Extraction

**Stage 1: spaCy NER**
- ORG (Organizations)
- PERSON (People)
- GPE (Geopolitical Entities)
- DATE (Dates)
- MONEY (Monetary values)

**Stage 2: Custom Regex Patterns**
```python
self.patterns = {
    'MONEY': [
        r'\$[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|M|B|K)?',
    ],
    'PERCENTAGE': [
        r'[\d.]+\s*%',
        r'[\d.]+\s*percent',
    ],
    'FISCAL_DATE': [
        r'(?:Q[1-4]|FY)\s*\'?\d{2,4}',
        r'fiscal\s+(?:year\s+)?\d{4}',
    ],
    'TICKER': [
        r'\b[A-Z]{1,5}\b(?=\s+(?:stock|shares|Inc|Corp))',
    ],
}
```

#### Usage Example

```python
from src.entity_extractor import EntityExtractor

ner = EntityExtractor()
entities = ner.extract("Apple Inc. reported Q4 2024 revenue of $94.9 billion.")

for ent in entities:
    print(f"{ent['type']}: {ent['text']}")
```

---

**Page 10**

---

### 5.4 Risk Factor Detector

**File:** `src/risk_detector.py`

#### Seven Risk Categories

| Category | Icon | Description | Example Patterns |
|----------|------|-------------|------------------|
| **Regulatory** | ⚖️ | Legal and compliance risks | `regulatory risk`, `SEC investigation`, `antitrust` |
| **Financial** | 💰 | Liquidity and credit risks | `material adverse effect`, `going concern`, `bankruptcy` |
| **Operational** | ⚙️ | Business operations risks | `supply chain disruption`, `cybersecurity threat`, `data breach` |
| **Market** | 📈 | Competition and economic risks | `competitive pressure`, `market volatility`, `pricing pressure` |
| **Pandemic** | 🦠 | Public health risks | `COVID-19`, `pandemic`, `public health crisis` |
| **Geopolitical** | 🌍 | Political and trade risks | `geopolitical tension`, `trade war`, `sanctions` |
| **Climate** | 🌡️ | Environmental risks | `climate change`, `carbon emissions`, `natural disaster` |

#### Severity Assessment

Each risk mention is assessed for severity based on context:

| Severity | Indicator Words | Weight |
|----------|-----------------|--------|
| 🔴 High | significantly, materially, substantially, severely | 3x |
| 🟡 Medium | moderately, considerably, notably | 2x |
| 🟢 Low | minor, slightly, limited, minimally | 1x |

#### Risk Score Calculation

```
Risk Score = (Σ severity_weight × count) / max_expected × 100
```

- Score 0-29: **Low** risk level
- Score 30-59: **Medium** risk level
- Score 60-100: **High** risk level

#### Usage Example

```python
from src.risk_detector import RiskDetector

detector = RiskDetector()
risks = detector.detect("The company faces significant regulatory risk...")
score, level = detector.get_risk_score(risks)
print(f"Risk Level: {level} (Score: {score:.0f}/100)")
```

---

**Page 11**

---

### 5.5 Financial Metrics Extractor

**File:** `src/metrics_extractor.py`

#### 14 Metric Types

| Metric Type | Icon | Pattern Description | Example Match |
|-------------|------|---------------------|---------------|
| Revenue | 💵 | Revenue/sales figures | "revenue of $65.6 billion" |
| Revenue Change | 📊 | Revenue increases/decreases | "revenue increased 11.2%" |
| Net Income | 💰 | Earnings/profit figures | "net income was $24.7 billion" |
| EPS | 📈 | Earnings per share | "EPS of $3.30" |
| Margin | 📉 | Gross/operating/net margins | "gross margin was 42.5%" |
| YoY Change | 📆 | Year-over-year comparisons | "15% increase YoY" |
| QoQ Change | 📅 | Quarter-over-quarter changes | "8% growth sequentially" |
| Guidance | 🔮 | Forward-looking projections | "expect revenue of $70B" |
| Cash Flow | 💸 | Operating/free cash flow | "free cash flow of $20B" |
| Debt | 📋 | Total/long-term debt | "total debt was $85B" |
| Assets | 🏦 | Total assets | "total assets of $320B" |
| Dividend | 💎 | Dividend per share | "dividend of $0.83/share" |
| Share Count | 📊 | Shares outstanding | "1.5B shares outstanding" |
| Headcount | 👥 | Employee count | "approximately 150,000 employees" |

### 5.6 Forward-Looking Statement Detector

**File:** `src/metrics_extractor.py` (ForwardLookingDetector class)

#### Confidence Level Indicators

**High Confidence:**
- will, expect, plan to, intend to, committed to, on track to, scheduled to

**Medium Confidence:**
- believe, anticipate, project, forecast, estimate, aim to, seek to, target

**Low Confidence:**
- may, could, might, possible, potentially, would, should, hope to

### 5.7 Extractive Summarizer

**File:** `src/summarizer.py`

#### Algorithm

1. Tokenize document into sentences using NLTK
2. Calculate TF-IDF scores for each sentence
3. Apply position weighting (first/last sentences boosted)
4. Select top N sentences by score
5. Return sentences in original document order

**Position Weighting:**
- First 3 sentences: 1.5x, 1.4x, 1.3x boost
- Last 2 sentences: 1.2x boost

---

**Page 12**

---

## 6. Running the Pipeline

### 6.1 Launch the Dashboard

After completing training (Section 4.3), launch the Gradio dashboard:

```bash
python app.py
```

Open your browser to: **http://localhost:7860**

### 6.2 Dashboard Features

#### Input Section
- **Text Area**: Paste financial document text (minimum 100 characters)
- **Sample Documents**: Load pre-included SEC filing excerpts with one click
- **Analysis Toggles**: Enable/disable specific analysis components

#### Analysis Options
- 📄 Document Classification
- 😊 Sentiment Analysis
- 🏷️ Named Entity Recognition
- ⚠️ Risk Detection
- 📊 Metrics & Forward-Looking
- 📝 Summarization

#### Output Sections
1. **Executive Summary Card**: Quick overview with document type, sentiment, risk level, key entities
2. **Highlighted Text**: Color-coded annotations for entities and risks
3. **Classification & Sentiment**: Probability charts and sentiment gauge
4. **Entities**: Distribution chart and sortable entity table
5. **Risk Analysis**: Risk score gauge and category breakdown
6. **Summary**: Extractive summary of key sentences
7. **Export**: Download complete analysis as JSON

### 6.3 Programmatic Usage

```python
from src.document_classifier import DocumentClassifier
from src.sentiment_analyzer import SentimentAnalyzer
from src.entity_extractor import EntityExtractor
from src.risk_detector import RiskDetector
from src.metrics_extractor import MetricsExtractor, ForwardLookingDetector
from src.summarizer import ExtractiveSummarizer

# Load text
text = open("data/sample_docs/apple_10k_2024.txt").read()

# Classification
classifier = DocumentClassifier()
classifier.load("models")
section_code, label, confidence = classifier.predict_with_label(text)

# Sentiment
sentiment = SentimentAnalyzer()
result = sentiment.analyze(text)

# Entities
ner = EntityExtractor()
entities = ner.extract(text)

# Risks
risk_detector = RiskDetector()
risks = risk_detector.detect(text)
score, level = risk_detector.get_risk_score(risks)

# Metrics
metrics = MetricsExtractor()
extracted = metrics.extract(text)

# Forward-Looking
fwd = ForwardLookingDetector()
statements = fwd.detect(text)

# Summary
summarizer = ExtractiveSummarizer()
summary = summarizer.summarize(text, num_sentences=5)
```

---

**Page 13**

---

## 7. Results and Visualizations

### 7.1 Document Classifier Performance

The Keras neural network classifier was trained on ~500 SEC 10-K sections and evaluated on a held-out test set.

**Model Configuration:**
- Input: TF-IDF vectors (3,000 features, unigrams + bigrams)
- Architecture: 256 → 128 → 64 → 20 neurons
- Regularization: BatchNormalization + Dropout (0.4, 0.3, 0.2)
- Training: Adam optimizer, early stopping (patience=3)

**Training Results:**
- Training completed in 10-15 epochs (with early stopping)
- Validation accuracy improved steadily during training
- Final model restored to best validation checkpoint

### 7.2 Sentiment Analysis Examples

| Text Sample | FinBERT Prediction | Confidence |
|-------------|-------------------|------------|
| "Revenue increased 15% year-over-year, beating expectations." | Positive | 94.2% |
| "The company reported a significant decline in profit margins." | Negative | 88.7% |
| "The Board declared a quarterly dividend of $0.25 per share." | Neutral | 76.3% |

### 7.3 Entity Extraction Results

Sample analysis of an Apple 10-K excerpt:

| Entity Type | Count | Examples |
|-------------|-------|----------|
| ORG | 12 | Apple Inc., NASDAQ, SEC |
| MONEY | 23 | $94.9 billion, $1.64 |
| PERCENTAGE | 15 | 6%, 12% |
| DATE | 8 | September 28, 2024 |
| FISCAL_DATE | 4 | Q4 2024, FY2024 |

### 7.4 Risk Detection Results

Sample risk analysis output:

| Risk Category | Mentions | Severity Distribution |
|---------------|----------|----------------------|
| Operational | 8 | High: 3, Medium: 4, Low: 1 |
| Market | 6 | High: 2, Medium: 3, Low: 1 |
| Regulatory | 4 | High: 1, Medium: 2, Low: 1 |
| Financial | 3 | High: 1, Medium: 1, Low: 1 |

**Overall Risk Score:** 45/100 (Medium)

### 7.5 Processing Performance

| Component | Typical Time |
|-----------|-------------|
| Classification | ~0.1s |
| Sentiment Analysis | ~1-2s |
| Entity Extraction | ~0.5s |
| Risk Detection | ~0.2s |
| Metrics Extraction | ~0.1s |
| Forward-Looking Detection | ~0.1s |
| Summarization | ~0.3s |
| **Total Pipeline** | **~3-5s** |

*Performance measured on CPU (Intel i7). GPU accelerates FinBERT sentiment analysis.*

---

**Page 14**

---

## 8. What Worked and What Did Not

### 8.1 What Worked Well

#### 1. Custom Keras Classifier
- **Success**: The TF-IDF + neural network approach achieved high accuracy for section classification
- **Key Factor**: Using bigrams in addition to unigrams captured important multi-word phrases
- **Benefit**: Fast inference (~0.1s per document) compared to transformer-based classifiers

#### 2. FinBERT for Financial Sentiment
- **Success**: Significantly outperformed general-purpose sentiment models on financial text
- **Key Factor**: Pre-trained on financial news and documents, understands domain terminology
- **Benefit**: Correctly interprets phrases like "revenue miss" as negative and "beat expectations" as positive

#### 3. Hybrid NER Approach
- **Success**: Combining spaCy with custom regex patterns captured both general and financial-specific entities
- **Key Factor**: Custom patterns for fiscal dates (Q1 2024), tickers (AAPL), and normalized monetary values
- **Benefit**: High recall for financial entities that spaCy alone would miss

#### 4. Risk Category Framework
- **Success**: The 7-category risk framework (regulatory, financial, operational, etc.) provided useful structure
- **Key Factor**: 70+ regex patterns with severity assessment based on context
- **Benefit**: Actionable risk scores that highlight areas of concern

#### 5. Gradio Dashboard
- **Success**: Provided an intuitive interface for non-technical users
- **Key Factor**: Real-time visualizations with Plotly (gauges, bar charts, tables)
- **Benefit**: JSON export enables integration with other tools

### 8.2 What Did Not Work as Expected

#### 1. Abstractive Summarization
- **Challenge**: Initially planned to use T5 for abstractive summarization
- **Issue**: T5 required GPU and was too slow on CPU (30+ seconds per document)
- **Solution**: Switched to TF-IDF-based extractive summarization (fast, no GPU needed)
- **Trade-off**: Summaries are extracted sentences, not generated prose

#### 2. Full Document Classification
- **Challenge**: Classifying very long documents (50,000+ characters)
- **Issue**: TF-IDF vectors become sparse and less discriminative for very long texts
- **Solution**: Truncate input to first 50,000 characters (covers most meaningful content)
- **Trade-off**: May miss important information at the end of very long documents

#### 3. FinBERT Token Limit
- **Challenge**: FinBERT has a 512 token limit
- **Issue**: Financial documents are typically much longer
- **Solution**: Truncate to ~400 words for sentiment analysis
- **Trade-off**: Sentiment reflects only the beginning of the document

#### 4. Complex Table Extraction
- **Challenge**: SEC filings contain many financial tables
- **Issue**: Regex patterns struggle with tabular data formats
- **Solution**: Focus on narrative text; skip dense numeric sections
- **Trade-off**: Some financial metrics embedded in tables are missed

---

**Page 15**

---

## 9. Lessons Learned

### 9.1 Technical Lessons

#### 1. Start Simple, Iterate
Started with complex transformer-based approaches for all components, but found that simpler methods (TF-IDF + neural network, extractive summarization) often provided sufficient accuracy with much faster performance.

#### 2. Domain-Specific Models Matter
FinBERT significantly outperformed general BERT/RoBERTa models on financial text. The lesson: always look for domain-specific pre-trained models when working in specialized domains.

#### 3. Regex Patterns Are Powerful
For structured financial data (monetary values, percentages, fiscal dates), well-crafted regex patterns often outperform ML-based approaches in both accuracy and speed.

#### 4. Modular Architecture Pays Off
Separating each NLP component into its own module made debugging easier and allowed swapping implementations without affecting the rest of the pipeline.

#### 5. CPU Compatibility Is Essential
Not all users have GPUs. Ensuring all components run on CPU (even if slower) made the project more accessible.

### 9.2 Project Management Lessons

#### 1. Scope Creep
The original plan included more components (topic modeling, abstractive summarization with T5). Learning to cut scope and focus on core functionality was essential.

#### 2. Data Quality
The HuggingFace SEC dataset was well-structured, but still required preprocessing (cleaning HTML artifacts, handling encoding issues, filtering very short sections).

#### 3. Documentation as You Go
Writing docstrings and comments during development (not after) saved significant time when building the dashboard and writing this report.

### 9.3 Future Improvements

1. **Section-Level Sentiment**: Analyze sentiment per section, not just overall document
2. **Named Entity Linking**: Link extracted entities to external knowledge bases (e.g., company databases)
3. **Comparative Analysis**: Compare current filing to previous years' filings
4. **Table Extraction**: Use specialized table extraction libraries (e.g., Camelot) for financial tables
5. **Fine-Tune FinBERT**: Fine-tune on SEC-specific data for even better performance

---

**Page 16**

---

## 10. Conclusion

This project successfully demonstrates a practical NLP pipeline for financial document analysis. The key contributions are:

1. **Custom Document Classifier**: A Keras neural network trained on SEC 10-K sections, achieving high accuracy with fast inference

2. **Multi-Component Pipeline**: Integration of seven NLP techniques (classification, sentiment, NER, risk detection, metrics extraction, forward-looking detection, summarization) into a unified system

3. **Interactive Dashboard**: A user-friendly Gradio interface with real-time visualizations and export capabilities

4. **Reproducible Implementation**: Well-documented code with Jupyter notebooks that allow colleagues to reproduce all results

The pipeline processes documents in 3-5 seconds on standard CPU hardware, making it practical for real-world use. All code is modular, documented, and available in the GitHub repository.

The project demonstrates that effective NLP systems can be built by combining pre-trained models (FinBERT, spaCy) with custom components (Keras classifier, regex patterns) tailored to the specific domain. The financial services industry offers many opportunities for NLP applications, and this project provides a foundation for further development.

---

## 11. References

1. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. *arXiv preprint arXiv:1908.10063*.

2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

3. Honnibal, M., & Montani, I. (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing.

4. U.S. Securities and Exchange Commission. EDGAR Database. https://www.sec.gov/edgar.shtml

5. Chollet, F. (2015). Keras. https://keras.io/

6. Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and Implementation*.

7. Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*.

8. Gradio Documentation. https://gradio.app/

---

**Page 17**

---

## 12. Appendix: Complete Code Listings

The complete source code is available in the GitHub repository:

**Repository URL:** https://github.com/shyamsridhar123/CSCI-S-89B-Final-Project

### Key Files

| File | Description | Lines |
|------|-------------|-------|
| `app.py` | Gradio dashboard application | ~600 |
| `src/document_classifier.py` | Keras neural network classifier | ~280 |
| `src/sentiment_analyzer.py` | FinBERT sentiment analysis | ~180 |
| `src/entity_extractor.py` | spaCy NER + custom patterns | ~220 |
| `src/risk_detector.py` | Risk factor detection | ~320 |
| `src/metrics_extractor.py` | Financial metrics + forward-looking | ~380 |
| `src/summarizer.py` | Extractive summarization | ~200 |

### Notebooks

| Notebook | Purpose |
|----------|---------|
| `00_setup_models.ipynb` | Pre-download models (optional) |
| `01_data_preparation.ipynb` | Download SEC data, create splits |
| `02_train_classifier.ipynb` | Train Keras classifier |
| `03_finbert_enhanced_detection.ipynb` | FinBERT-enhanced risk/forward-looking demo |

---

## Video Presentation

**Link:** https://youtu.be/CKjtqYBO6cc

The video presentation (7-15 minutes) includes:
- Project overview and problem statement
- Live demonstration of the dashboard
- Code walkthrough of key components
- Discussion of results and lessons learned

---

**End of Report**

---

*This report was prepared for CSCI S-89B Introduction to Natural Language Processing at Harvard Extension School.*
