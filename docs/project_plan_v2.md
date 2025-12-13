# Financial Document Intelligence Pipeline - Streamlined Project Plan (v2)

## Project Overview

**Course**: CSCI S-89B Introduction to Natural Language Processing  
**Student**: Shyam Sridhar  
**Project Title**: Financial Document Intelligence Pipeline with Interactive Dashboard  
**Topic Category**: Combining Multiple NLP Techniques (Classification, Sentiment Analysis, NER)

---

## Key Changes from Original Plan

| Aspect | Original | Streamlined |
|--------|----------|-------------|
| NLP Components | 5 (Classification, NER, Sentiment, Risk, Summarization) | 3 (Classification, Sentiment, NER) |
| Summarization | T5 Abstractive Model | Extractive (TextRank/NLTK) |
| NER Approach | FinBERT + Complex Patterns | spaCy + Regex Patterns |
| Dashboard Charts | 5+ interactive visualizations | 3 core visualizations |
| Custom Training | None (all pre-trained) | Keras classifier (demonstrates TF skills) |
| Processing Time | 7-12 sec/doc | 2-5 sec/doc |

---

## Problem Statement

Develop a practical NLP system that automatically processes financial documents (SEC filings, earnings reports) to:
1. **Classify** document types using a custom-trained Keras model
2. **Analyze sentiment** using FinBERT for financial context
3. **Extract entities** using spaCy + custom financial patterns
4. **Generate summaries** using extractive methods (fast, no GPU needed)
5. **Visualize results** through an interactive Gradio dashboard

---

## Dataset

### Source
**HuggingFace Dataset**: [`JanosAudran/financial-reports-sec`](https://huggingface.co/datasets/JanosAudran/financial-reports-sec)

**Available Configurations**:
- `large_full` - Complete SEC filings with full text (use for training)
- `large_lite` - Lighter version with metadata
- `small_full` - Smaller subset (use for development/testing)

**Document Types Available**: 10-K, 10-Q, 8-K (aligns with classification targets)

### Dataset Size (Manageable)
- Training: 500 documents
- Validation: 100 documents  
- Test: 100 documents
- **Total**: 700 documents

*Sample documents will be included in repository for demo purposes.*

---

## Technical Architecture (Simplified)

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Financial Document                     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: Preprocessing & Classification             │
│                                                                   │
│  • Text cleaning (regex, normalization)                          │
│  • TF-IDF Vectorization                                          │
│  • Keras Neural Network Classifier ← CUSTOM TRAINED              │
│                                                                   │
│  Output: Document Type (10-K, 10-Q, 8-K, Earnings)               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: Sentiment Analysis                     │
│                                                                   │
│  • FinBERT (ProsusAI/finbert) - pre-trained                     │
│  • Financial-specific sentiment (positive/negative/neutral)      │
│  • Confidence scores                                             │
│                                                                   │
│  Output: Sentiment label + score                                 │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: Named Entity Recognition                   │
│                                                                   │
│  • spaCy (en_core_web_sm) for general NER                       │
│  • Custom regex patterns for:                                    │
│    - Monetary values ($X million/billion)                        │
│    - Percentages                                                 │
│    - Dates (fiscal quarters, years)                              │
│                                                                   │
│  Output: List of entities with types                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 4: Extractive Summarization                   │
│                                                                   │
│  • NLTK sentence tokenization                                    │
│  • TF-IDF sentence scoring                                       │
│  • Top-N sentence extraction                                     │
│                                                                   │
│  Output: 3-5 key sentences as summary                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GRADIO DASHBOARD                              │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Sentiment   │  │   Entity     │  │   Summary    │           │
│  │    Gauge     │  │    Chart     │  │   Display    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Components

### 1. Document Classifier (CUSTOM KERAS MODEL - Key Deliverable)

This demonstrates proficiency with Keras/TensorFlow as required.

```python
# src/document_classifier.py
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class DocumentClassifier:
    """
    Custom Keras neural network for document classification.
    Trained on SEC filing types: 10-K, 10-Q, 8-K, Earnings
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.classes = ['10-K', '10-Q', '8-K', 'Earnings']
    
    def build_model(self, input_dim):
        """Build a simple feedforward neural network"""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, texts, labels, epochs=10, validation_split=0.2):
        """Train the classifier"""
        X = self.vectorizer.fit_transform(texts).toarray()
        self.model = self.build_model(X.shape[1])
        
        history = self.model.fit(
            X, labels,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=32,
            verbose=1
        )
        return history
    
    def predict(self, text):
        """Classify a single document"""
        X = self.vectorizer.transform([text]).toarray()
        probs = self.model.predict(X, verbose=0)[0]
        predicted_class = self.classes[np.argmax(probs)]
        confidence = float(np.max(probs))
        return predicted_class, confidence
    
    def save(self, path):
        """Save model and vectorizer"""
        self.model.save(f"{path}/classifier_model.keras")
        import joblib
        joblib.dump(self.vectorizer, f"{path}/vectorizer.joblib")
    
    def load(self, path):
        """Load saved model and vectorizer"""
        self.model = keras.models.load_model(f"{path}/classifier_model.keras")
        import joblib
        self.vectorizer = joblib.load(f"{path}/vectorizer.joblib")
```

### 2. Sentiment Analyzer (FinBERT - Pre-trained)

```python
# src/sentiment_analyzer.py
from transformers import pipeline
import numpy as np

class SentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT.
    Returns: positive, negative, or neutral with confidence.
    """
    
    def __init__(self):
        print("Loading FinBERT model (this may take 30-60 seconds)...")
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
        self.max_length = 512  # FinBERT token limit
    
    def analyze(self, text):
        """Analyze sentiment of text"""
        # Truncate to max length if needed
        words = text.split()[:400]  # Rough token approximation
        truncated = ' '.join(words)
        
        result = self.pipeline(truncated)[0]
        return {
            'label': result['label'],
            'score': round(result['score'], 4),
            'interpretation': self._interpret(result['label'], result['score'])
        }
    
    def _interpret(self, label, score):
        """Human-readable interpretation"""
        strength = "strongly" if score > 0.8 else "moderately" if score > 0.6 else "slightly"
        return f"Document is {strength} {label.lower()} (confidence: {score:.1%})"
```

### 3. Entity Extractor (spaCy + Custom Patterns)

```python
# src/entity_extractor.py
import spacy
import re
from collections import Counter

class EntityExtractor:
    """
    Extract named entities using spaCy + custom financial patterns.
    """
    
    def __init__(self):
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Custom patterns for financial entities
        self.patterns = {
            'MONEY': r'\$[\d,]+(?:\.\d{2})?\s*(?:million|billion|trillion|M|B|K)?',
            'PERCENTAGE': r'[\d.]+\s*%|[\d.]+\s*percent',
            'FISCAL_DATE': r'(?:Q[1-4]|FY)\s*\d{4}|fiscal\s+(?:year\s+)?\d{4}',
            'TICKER': r'\b[A-Z]{2,5}\b(?=\s+(?:stock|shares|Inc|Corp))'
        }
    
    def extract(self, text):
        """Extract all entities from text"""
        entities = []
        
        # spaCy NER
        doc = self.nlp(text[:100000])  # Limit for performance
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'GPE', 'DATE', 'MONEY']:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'source': 'spacy'
                })
        
        # Custom pattern matching
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match,
                    'type': entity_type,
                    'source': 'pattern'
                })
        
        return self._deduplicate(entities)
    
    def _deduplicate(self, entities):
        """Remove duplicate entities"""
        seen = set()
        unique = []
        for ent in entities:
            key = (ent['text'].lower(), ent['type'])
            if key not in seen:
                seen.add(key)
                unique.append(ent)
        return unique
    
    def get_summary(self, entities):
        """Get entity type counts for visualization"""
        type_counts = Counter(e['type'] for e in entities)
        return dict(type_counts)
```

### 4. Extractive Summarizer (NLTK-based - Fast, No GPU)

```python
# src/summarizer.py
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class ExtractiveSummarizer:
    """
    Fast extractive summarization using TF-IDF sentence scoring.
    No GPU required - runs quickly on CPU.
    """
    
    def __init__(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    
    def summarize(self, text, num_sentences=5):
        """Extract top N most important sentences"""
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Score sentences using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Sum TF-IDF scores for each sentence
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get top N sentence indices (preserve original order)
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices = sorted(top_indices)
        
        # Build summary
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
```

### 5. Gradio Dashboard (Simplified)

```python
# app.py
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px

from src.document_classifier import DocumentClassifier
from src.sentiment_analyzer import SentimentAnalyzer
from src.entity_extractor import EntityExtractor
from src.summarizer import ExtractiveSummarizer

class FinancialDashboard:
    def __init__(self):
        print("Initializing Financial Document Intelligence Pipeline...")
        self.classifier = DocumentClassifier()
        self.classifier.load("models")  # Load pre-trained
        self.sentiment = SentimentAnalyzer()
        self.ner = EntityExtractor()
        self.summarizer = ExtractiveSummarizer()
        print("Pipeline ready!")
    
    def analyze(self, text):
        """Run full analysis pipeline"""
        # Classification
        doc_type, doc_confidence = self.classifier.predict(text)
        
        # Sentiment
        sentiment_result = self.sentiment.analyze(text)
        
        # NER
        entities = self.ner.extract(text)
        entity_summary = self.ner.get_summary(entities)
        
        # Summary
        summary = self.summarizer.summarize(text, num_sentences=5)
        
        return {
            'classification': {'type': doc_type, 'confidence': doc_confidence},
            'sentiment': sentiment_result,
            'entities': entities,
            'entity_summary': entity_summary,
            'summary': summary
        }
    
    def create_sentiment_gauge(self, sentiment_result):
        """Create sentiment gauge visualization"""
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        # Map to 0-100 scale
        if label == 'positive':
            value = 50 + (score * 50)
        elif label == 'negative':
            value = 50 - (score * 50)
        else:
            value = 50
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': f"Sentiment: {label.upper()}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 33], 'color': "red"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "green"}
                ]
            }
        ))
        fig.update_layout(height=300)
        return fig
    
    def create_entity_chart(self, entity_summary):
        """Create entity distribution bar chart"""
        if not entity_summary:
            return None
        
        fig = px.bar(
            x=list(entity_summary.keys()),
            y=list(entity_summary.values()),
            labels={'x': 'Entity Type', 'y': 'Count'},
            title='Extracted Entities by Type',
            color=list(entity_summary.values()),
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=300, showlegend=False)
        return fig

def create_app():
    dashboard = FinancialDashboard()
    
    def process_document(text):
        if not text or len(text.strip()) < 100:
            return "Please enter at least 100 characters", None, None, "", ""
        
        results = dashboard.analyze(text)
        
        # Create visualizations
        sentiment_fig = dashboard.create_sentiment_gauge(results['sentiment'])
        entity_fig = dashboard.create_entity_chart(results['entity_summary'])
        
        # Format classification result
        classification = f"**Document Type**: {results['classification']['type']}\n"
        classification += f"**Confidence**: {results['classification']['confidence']:.1%}"
        
        # Format entity list
        entity_text = "### Top Entities Found:\n"
        for ent in results['entities'][:15]:
            entity_text += f"- **{ent['type']}**: {ent['text']}\n"
        
        return (
            classification,
            sentiment_fig,
            entity_fig,
            results['summary'],
            entity_text
        )
    
    with gr.Blocks(title="Financial Document Intelligence") as app:
        gr.Markdown("# 📊 Financial Document Intelligence Pipeline")
        gr.Markdown("Analyze SEC filings and financial documents using NLP")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Paste Financial Document Text",
                    placeholder="Paste SEC filing, earnings report, or financial news...",
                    lines=15
                )
                analyze_btn = gr.Button("🔍 Analyze Document", variant="primary")
            
            with gr.Column(scale=1):
                classification_output = gr.Markdown(label="Classification")
        
        with gr.Row():
            sentiment_plot = gr.Plot(label="Sentiment Analysis")
            entity_plot = gr.Plot(label="Entity Distribution")
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Textbox(label="Executive Summary", lines=5)
            with gr.Column():
                entities_output = gr.Markdown(label="Extracted Entities")
        
        analyze_btn.click(
            fn=process_document,
            inputs=[input_text],
            outputs=[classification_output, sentiment_plot, entity_plot, 
                    summary_output, entities_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
```

---

## Project Structure (Simplified)

```
Project/
├── data/
│   ├── raw/                  # Original SEC filings
│   ├── processed/            # Cleaned data
│   └── sample_docs/          # Demo documents
│
├── models/
│   ├── classifier_model.keras   # Trained Keras model
│   └── vectorizer.joblib        # TF-IDF vectorizer
│
├── src/
│   ├── __init__.py
│   ├── document_classifier.py   # Custom Keras classifier
│   ├── sentiment_analyzer.py    # FinBERT wrapper
│   ├── entity_extractor.py      # spaCy + patterns
│   └── summarizer.py            # Extractive summarization
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_train_classifier.ipynb    # KEY: Shows Keras training
│   └── 03_full_pipeline_demo.ipynb
│
├── app.py                    # Gradio dashboard
├── requirements.txt
├── README.md
└── docs/
    ├── final_report.pdf
    └── presentation.pptx
```

---

## Requirements (Minimal)

```
# Core
tensorflow>=2.10.0
transformers>=4.30.0
torch>=2.0.0
gradio>=4.0.0

# NLP
spacy>=3.5.0
nltk>=3.8

# Data & Viz
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.14.0

# Utilities
joblib
python-dotenv
```

---

## Timeline (Realistic)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| **Week 1** | Data collection, preprocessing, train classifier | `02_train_classifier.ipynb` with trained model |
| **Week 2** | Integrate sentiment + NER + summarizer | All `src/` modules working |
| **Week 3** | Build Gradio dashboard, testing | Working `app.py` |
| **Week 4** | Documentation, slides, video | Report, slides, video |

---

## Performance Expectations (Realistic)

| Component | Time per Document | Memory |
|-----------|-------------------|--------|
| Classification | 0.1 sec | ~100MB |
| Sentiment (FinBERT) | 1-2 sec | ~1GB |
| NER (spaCy) | 0.5 sec | ~200MB |
| Summarization | 0.3 sec | ~50MB |
| **Total** | **2-4 sec** | **~1.5GB** |

---

## Deliverables Checklist

- [ ] **Working Code** (7 points)
  - [ ] Custom Keras classifier trained and saved
  - [ ] All 4 pipeline modules functional
  - [ ] Gradio dashboard running locally
  - [ ] Sample documents for demo

- [ ] **Report** (part of 7 points)
  - [ ] Abstract and problem statement
  - [ ] Installation instructions (reproducible)
  - [ ] Results and visualizations
  - [ ] Lessons learned

- [ ] **Slides** (3 points)
  - [ ] 10-15 slides with white background
  - [ ] Screenshots of working demo
  - [ ] Pros/cons analysis

- [ ] **Video** (3 points)
  - [ ] 7-15 minute walkthrough
  - [ ] Live demo of pipeline
  - [ ] Code explanation

---

## Key Differentiators

1. **Custom Keras Model** - Demonstrates TensorFlow proficiency as required
2. **Fast Processing** - 2-4 seconds per document on CPU
3. **Practical Application** - Real-world financial analysis use case
4. **Clean Architecture** - Modular, documented, reproducible
5. **Interactive Demo** - Gradio dashboard for live demonstration

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| FinBERT too slow | Pre-load model; process in batches |
| Dataset loading issues | Use `small_full` config for dev; `large_full` for training |
| GPU not available | All components run on CPU |
| Time constraints | Core pipeline works; dashboard is bonus |
