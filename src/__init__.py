"""
Financial Document Intelligence Pipeline
CSCI S-89B Final Project - Shyam Sridhar

Modules:
- document_classifier: Custom Keras neural network for SEC filing classification
- sentiment_analyzer: FinBERT-based financial sentiment analysis
- entity_extractor: spaCy + custom patterns for financial NER
- summarizer: Extractive summarization using TF-IDF scoring
"""

from .document_classifier import DocumentClassifier
from .sentiment_analyzer import SentimentAnalyzer
from .entity_extractor import EntityExtractor
from .summarizer import ExtractiveSummarizer

__version__ = "1.0.0"
__author__ = "Shyam Sridhar"

__all__ = [
    "DocumentClassifier",
    "SentimentAnalyzer", 
    "EntityExtractor",
    "ExtractiveSummarizer"
]
