# Financial Document Intelligence Pipeline

**CSCI S-89B Introduction to Natural Language Processing — Final Project**  
**Author:** Shyam Sridhar  
**Harvard Extension School**

---

## 📋 Abstract

This project presents a comprehensive Natural Language Processing (NLP) pipeline for analyzing SEC filings and financial documents. The system integrates multiple NLP techniques including document classification using a custom Keras neural network, sentiment analysis with FinBERT, named entity recognition via spaCy, risk factor detection with optional FinBERT-enhanced severity scoring, financial metrics extraction, forward-looking statement detection, and extractive summarization. All components are accessible through an interactive Gradio dashboard with real-time visualizations and downloadable reports.

The pipeline demonstrates practical applications of NLP in the financial domain, addressing the challenge of extracting actionable insights from complex regulatory filings. The document classifier achieves high accuracy on 20 SEC 10-K section types, while FinBERT provides domain-specific sentiment analysis trained on financial text. The integrated analysis provides investors, analysts, and compliance professionals with a unified tool for financial document understanding.

---

## 🎯 Problem Statement

Financial analysts and investors face significant challenges when analyzing SEC filings:

1. **Volume**: Thousands of 10-K, 10-Q, and 8-K filings are submitted annually
2. **Complexity**: Documents contain dense legal and financial language
3. **Time**: Manual analysis of a single 10-K can take hours
4. **Consistency**: Human analysis varies in quality and focus

This pipeline solves these problems by providing automated, consistent, and comprehensive analysis of financial documents in seconds.

---

## ✨ Features

| Component | Technology | Description |
|-----------|------------|-------------|
| **Document Classification** | Custom Keras Neural Network + TF-IDF | Classifies SEC 10-K sections (20 classes) with human-readable labels |
| **Sentiment Analysis** | FinBERT (`ProsusAI/finbert`) | Financial-domain sentiment (positive/negative/neutral) with confidence scores |
| **Named Entity Recognition** | spaCy + Custom Regex Patterns | Extracts ORG, MONEY, PERCENTAGE, FISCAL_DATE, TICKER, and more |
| **Risk Factor Detection** | Pattern-based + FinBERT (optional) | Identifies regulatory, financial, operational, market, pandemic, geopolitical, and climate risks with sentiment-based severity |
| **Financial Metrics Extraction** | Regex patterns | Extracts revenue, EPS, margins, YoY changes, guidance, dividends |
| **Forward-Looking Statements** | Keyword detection + FinBERT (optional) | Detects predictive language with high/medium/low confidence and sentiment analysis |
| **Extractive Summarization** | TF-IDF sentence scoring + position weighting | Extracts key sentences based on term frequency (fast, no GPU required) |
| **Interactive Dashboard** | Gradio + Plotly | Real-time analysis with visualizations and downloadable JSON reports |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ (tested with 3.11)
- 4GB+ RAM (FinBERT requires ~1GB)
- GPU optional (CPU works fine, GPU accelerates FinBERT)

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd NLPFinalProject

# Using conda (recommended)
conda create -n nlp-pipeline python=3.11
conda activate nlp-pipeline

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### 2. Run Notebooks (Required Before Dashboard)

The classifier must be trained before using the dashboard. **Run notebooks in order:**

| Step | Notebook | Purpose | Time |
|------|----------|---------|------|
| 0 | `00_setup_models.ipynb` | (Optional) Pre-download all models | ~2-5 min |
| 1 | `01_data_preparation.ipynb` | Download SEC data, create train/val/test splits | ~5-10 min |
| 2 | `02_train_classifier.ipynb` | Train Keras neural network classifier | ~2-5 min |
| 3 | `03_finbert_enhanced_detection.ipynb` | (Optional) FinBERT-enhanced risk detection demo | ~2-3 min |

```bash
# Run data preparation first
jupyter notebook notebooks/01_data_preparation.ipynb
# Execute all cells

# Then train the classifier
jupyter notebook notebooks/02_train_classifier.ipynb
# Execute all cells
```

**Training Output:**
- `models/classifier_model.keras` — Keras neural network
- `models/vectorizer.joblib` — TF-IDF vectorizer
- `models/label_encoder.joblib` — Label encoder
- `models/classes.joblib` — Class definitions

### 3. (Optional) Explore FinBERT-Enhanced Detection

```bash
jupyter notebook notebooks/03_finbert_enhanced_detection.ipynb
```

This notebook demonstrates how FinBERT can enhance risk detection and forward-looking statement analysis by adding sentiment-based severity scoring.

### 4. Launch the Dashboard

```bash
python app.py
```

Open your browser to **http://localhost:7860**

---

## 📁 Project Structure

```
NLPFinalProject/
├── app.py                        # Main Gradio dashboard application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
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
├── notebooks/                    # Jupyter notebooks (see notebooks/README.md for details)
│   ├── README.md                 # 📋 Notebook execution guide
│   ├── 00_setup_models.ipynb     # Pre-download models (optional)
│   ├── 01_data_preparation.ipynb # Data download and preprocessing
│   ├── 02_train_classifier.ipynb # Model training and evaluation
│   └── 03_finbert_enhanced_detection.ipynb  # FinBERT hybrid approach for risk/forward-looking
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
│       ├── sample_section_1.txt
│       ├── sample_section_1A.txt
│       └── sample_section_7.txt
│
└── docs/                         # Documentation
    ├── description.md            # Project requirements
    ├── enhancement_plan.md       # Future improvements
    ├── training_history.png      # Training curves
    └── confusion_matrix.png      # Model evaluation
```

---

## 🔧 Component Details

### 1. Document Classifier (Keras Neural Network)

Classifies documents into 20 SEC 10-K section types:

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

**Architecture:**
```
Input: TF-IDF Vector (3000 features, unigrams + bigrams)
    ↓
Dense(256, ReLU) + BatchNorm + Dropout(0.4)
    ↓
Dense(128, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense(64, ReLU) + Dropout(0.2)
    ↓
Dense(20, Softmax) → Section Classification
```

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Early Stopping: Patience=3
- Learning Rate Reduction: Factor=0.5, Patience=2

### 2. Sentiment Analyzer (FinBERT)

Uses [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) for financial sentiment:
- **Positive**: Optimistic language, growth mentions
- **Neutral**: Factual statements, standard disclosures
- **Negative**: Risks, declines, concerns

### 3. Entity Extractor (spaCy + Custom Patterns)

Extracts entities using spaCy's `en_core_web_sm` plus custom regex:

| Entity Type | Examples |
|-------------|----------|
| `ORG` | Apple Inc., Microsoft Corporation |
| `PERSON` | Tim Cook, Satya Nadella |
| `MONEY` | $65.6 billion, $3.30 |
| `PERCENTAGE` | 16%, 71.2% |
| `DATE` | October 30, 2024, fiscal 2024 |
| `FISCAL_DATE` | Q1 2024, fiscal year 2024 |
| `TICKER` | AAPL, MSFT |
| `FINANCIAL_METRIC` | revenue, EPS, gross margin |

### 4. Risk Detector (7 Categories with 70+ Patterns)

The risk detector scans documents for risk-related language using 70+ regex patterns across 7 categories:

| Category | Icon | Description | Pattern Examples |
|----------|------|-------------|------------------|
| **Regulatory** | ⚖️ | Legal, compliance, and government regulation risks | `regulatory risk`, `SEC investigation`, `antitrust`, `compliance failure`, `legislative change` |
| **Financial** | 💰 | Liquidity, credit, and financial health risks | `material adverse effect`, `liquidity risk`, `debt covenant`, `impairment charge`, `going concern`, `bankruptcy` |
| **Operational** | ⚙️ | Business operations, supply chain, and security risks | `supply chain disruption`, `cybersecurity threat`, `data breach`, `key personnel loss`, `product recall` |
| **Market** | 📈 | Competition, economic, and market condition risks | `competitive pressure`, `market volatility`, `economic recession`, `foreign exchange risk`, `pricing pressure` |
| **Pandemic** | 🦠 | Public health and pandemic-related risks | `COVID-19`, `pandemic`, `public health crisis`, `quarantine`, `remote working` |
| **Geopolitical** | 🌍 | Political, trade, and international risks | `geopolitical tension`, `trade war`, `tariff`, `sanctions`, `political instability`, `terrorism` |
| **Climate** | 🌡️ | Environmental, climate, and ESG risks | `climate change`, `carbon emissions`, `natural disaster`, `ESG requirement`, `sustainability risk` |

**Severity Assessment:**

Each risk mention is assessed for severity based on surrounding context:

| Severity | Indicator Words | Weight |
|----------|----------------|--------|
| 🔴 **High** | significantly, materially, substantially, severely, critically, major | 3x |
| 🟡 **Medium** | moderately, considerably, notably | 2x |
| 🟢 **Low** | minor, slightly, limited, minimally | 1x |

**Risk Score Calculation:**
```
Risk Score = (Σ severity_weight × count) / max_expected × 100
```
- Score 0-29: **Low** risk level
- Score 30-59: **Medium** risk level  
- Score 60-100: **High** risk level

### 5. Metrics Extractor (14 Financial Metric Types)

Extracts and categorizes quantitative financial data using specialized regex patterns:

| Metric Type | Icon | Pattern Description | Example Match |
|-------------|------|---------------------|---------------|
| **Revenue** | 💵 | Revenue/sales figures | "revenue of $65.6 billion" |
| **Revenue Change** | 📊 | Revenue increases/decreases | "revenue increased by $166.4 million or 11.2%" |
| **Net Income** | 💰 | Earnings/profit figures | "net income was $24.7 billion" |
| **EPS** | 📈 | Earnings per share | "EPS of $3.30" |
| **Margin** | 📉 | Gross/operating/net margins | "gross margin was 42.5%" |
| **YoY Change** | 📆 | Year-over-year comparisons | "15% increase year-over-year" |
| **QoQ Change** | 📅 | Quarter-over-quarter changes | "8% growth sequentially" |
| **Guidance** | 🔮 | Forward-looking projections | "expect revenue of $70 billion" |
| **Cash Flow** | 💸 | Operating/free cash flow | "free cash flow of $20 billion" |
| **Debt** | 📋 | Total/long-term debt | "total debt was $85 billion" |
| **Assets** | 🏦 | Total assets | "total assets of $320 billion" |
| **Dividend** | 💎 | Dividend per share | "dividend of $0.83 per share" |
| **Share Count** | 📊 | Shares outstanding | "1.5 billion shares outstanding" |
| **Headcount** | 👥 | Employee count | "approximately 150,000 employees" |

**Value Normalization:**

Monetary values are normalized to a common scale:
- "K" / "thousand" → ×1,000
- "M" / "million" → ×1,000,000
- "B" / "billion" → ×1,000,000,000

**Change Direction Detection:**

The extractor identifies whether metrics represent increases or decreases:
- **Increase indicators**: increased, grew, rose, higher, growth
- **Decrease indicators**: decreased, declined, fell, lower, reduction

### 6. Forward-Looking Statement Detector

Identifies predictive and forward-looking language in SEC filings, categorizing statements by confidence level based on the strength of the language used:

**High Confidence Indicators** 🟢
| Phrase | Interpretation |
|--------|----------------|
| "will" | Definite commitment |
| "expect" | Strong anticipation |
| "plan to" | Confirmed intention |
| "intend to" | Clear objective |
| "committed to" | Firm pledge |
| "on track to" | Progress confirmation |
| "scheduled to" | Timed commitment |
| "set to" | Imminent action |

**Medium Confidence Indicators** 🟡
| Phrase | Interpretation |
|--------|----------------|
| "believe" | Reasoned opinion |
| "anticipate" | Expected outcome |
| "project" | Calculated estimate |
| "forecast" | Predictive analysis |
| "estimate" | Approximation |
| "aim to" | Goal-oriented |
| "seek to" | Attempt planned |
| "target" | Objective set |

**Low Confidence Indicators** 🔴
| Phrase | Interpretation |
|--------|----------------|
| "may" | Possibility |
| "could" | Conditional |
| "might" | Uncertain |
| "possible" | Speculative |
| "potentially" | Tentative |
| "would" | Hypothetical |
| "should" | Advisory |
| "hope to" | Aspirational |

**Processing:**
1. Text is tokenized into sentences using NLTK
2. Each sentence is scanned for indicator phrases
3. First matching indicator determines confidence level
4. Results include the sentence, trigger phrase, and confidence rating

### 7. Extractive Summarizer

Generates summaries by extracting the most important sentences (not generative/abstractive):

**How it works:**
1. Tokenizes document into sentences using NLTK
2. Calculates TF-IDF scores for each sentence
3. Applies position weighting (first/last sentences boosted)
4. Selects top N sentences by score
5. Returns sentences in original document order

**Limitations:**
- Extracts existing sentences verbatim (no paraphrasing)
- Quality depends on document structure
- Not suitable for highly technical documents with short sentences

**Note:** For abstractive summarization (generating new text), consider models like BART or T5, which require more GPU memory.

---

## 📊 Dashboard Features

The Gradio dashboard provides:

1. **Input Options**
   - Paste text directly into the text area
   - Load sample SEC filing excerpts with one click

2. **Analysis Toggles**
   - 📄 Document Classification
   - 😊 Sentiment Analysis
   - 🏷️ Named Entity Recognition
   - ⚠️ Risk Detection
   - 📊 Metrics & Forward-Looking
   - 📝 Summarization

3. **Results Display**
   - **Executive Summary Card**: Quick overview with document type, sentiment, risk level, key entities
   - **Highlighted Text**: Color-coded annotations for entities, risks, and forward-looking statements
   - **Classification & Sentiment**: Probability charts and sentiment gauge
   - **Entities**: Distribution chart and sortable entity table
   - **Risk Analysis**: Risk score gauge and category breakdown
   - **Summary**: Extractive summary of key sentences
   - **Export**: Download complete analysis as JSON

4. **Visualizations** (Plotly)
   - Sentiment gauge (positive/neutral/negative)
   - Classification probability bar chart
   - Entity type distribution pie chart
   - Risk category breakdown
   - Risk score gauge (0-100)

---

## ⚡ Performance

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

*Performance on CPU. GPU accelerates sentiment analysis.*

---

## 📊 Dataset

**Source:** [JanosAudran/financial-reports-sec](https://huggingface.co/datasets/JanosAudran/financial-reports-sec)

The dataset contains SEC 10-K filings parsed by section. The notebook `01_data_preparation.ipynb` downloads and preprocesses the data for training.

---

## 💻 Programmatic Usage

```python
from src.document_classifier import DocumentClassifier
from src.sentiment_analyzer import SentimentAnalyzer
from src.entity_extractor import EntityExtractor
from src.risk_detector import RiskDetector
from src.metrics_extractor import MetricsExtractor, ForwardLookingDetector
from src.summarizer import ExtractiveSummarizer

# Document Classification
classifier = DocumentClassifier()
classifier.load("models")
section_code, readable_label, confidence = classifier.predict_with_label(text)
print(f"Section: {readable_label} ({confidence:.1%})")

# All class probabilities
probs = classifier.predict_proba(text)

# Sentiment Analysis
sentiment = SentimentAnalyzer()
result = sentiment.analyze(text)
print(f"Sentiment: {result['label']} ({result['score']:.1%})")

# Entity Extraction
ner = EntityExtractor()
entities = ner.extract(text)
summary = ner.get_summary(entities)

# Risk Detection (keyword-based)
risk_detector = RiskDetector()
risks = risk_detector.detect(text)
score, level = risk_detector.get_risk_score(risks)
print(f"Risk Level: {level} (Score: {score:.0f}/100)")

# Risk Detection (FinBERT-enhanced for better severity scoring)
# Requires: from transformers import pipeline
# finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
# risk_detector_enhanced = RiskDetector(finbert_pipeline=finbert)

# Financial Metrics
metrics = MetricsExtractor()
extracted = metrics.extract(text)

# Forward-Looking Statements (keyword-based)
fwd = ForwardLookingDetector()
statements = fwd.detect(text)

# Forward-Looking Statements (FinBERT-enhanced)
# fwd_enhanced = ForwardLookingDetector(finbert_pipeline=finbert)

# Summarization
summarizer = ExtractiveSummarizer()
summary = summarizer.summarize(text, num_sentences=5)
```

---

## 🐛 Troubleshooting

### CuDNN Version Mismatch Error

If you see:
```
Loaded runtime CuDNN library: 9.1.0 but source was compiled with: 9.3.0
DNN library initialization failed
```

**Fix 1 - Force CPU mode:**
```bash
export CUDA_VISIBLE_DEVICES=""
python app.py
```

**Fix 2 - Update CuDNN:**
```bash
conda install cudnn=9.3
```

### TensorFlow Import Hangs

If importing TensorFlow hangs in Jupyter:
1. Restart the kernel
2. Add to the first cell:
   ```python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
   ```

### spaCy Model Not Found

```bash
python -m spacy download en_core_web_sm
```

### FinBERT Download Issues

The model downloads automatically on first use (~500MB). Ensure internet connectivity.

---

## 📦 Dependencies

```
# Core ML/DL
tensorflow>=2.10.0
torch>=2.0.0
transformers>=4.30.0

# NLP
spacy>=3.5.0
nltk>=3.8

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
datasets>=2.14.0

# Dashboard
gradio>=6.0.0
plotly>=5.14.0

# File Processing
PyPDF2>=3.0.0

# Utilities
joblib>=1.3.0
```

---

##  References

- [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
- [spaCy: Industrial-Strength NLP](https://spacy.io/)
- [SEC EDGAR Database](https://www.sec.gov/edgar.shtml)
- [Keras Documentation](https://keras.io/)
- [Gradio Documentation](https://gradio.app/)

---

## 📄 License

This project was developed for educational purposes as part of CSCI S-89B at Harvard Extension School.

---

## 👤 Author

**Shyam Sridhar**  
CSCI S-89B Introduction to Natural Language Processing  
Harvard Extension School

---

## 🎥 Video Presentation

[Link to Video Presentation] *(Add YouTube/Zoom link here)*
