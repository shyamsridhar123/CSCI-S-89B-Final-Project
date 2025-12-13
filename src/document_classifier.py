"""
Document Classifier Module
Custom Keras neural network for SEC filing classification.

This module demonstrates proficiency with Keras/TensorFlow as required
for the CSCI S-89B final project.

Author: Shyam Sridhar
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Optional, Dict
import joblib
import re


class DocumentClassifier:
    """
    Custom Keras neural network for SEC document classification.
    
    Classifies financial documents into SEC 10-K sections.
    Uses TF-IDF vectorization + feedforward neural network.
    
    Attributes:
        vectorizer: TF-IDF vectorizer for text feature extraction
        model: Keras Sequential model
        label_encoder: Sklearn LabelEncoder for class labels
        classes: List of document type classes
    """
    
    # Human-readable section labels
    SECTION_LABELS = {
        'section_1': 'Item 1 - Business Overview',
        'section_1A': 'Item 1A - Risk Factors',
        'section_1B': 'Item 1B - Unresolved Staff Comments',
        'section_2': 'Item 2 - Properties',
        'section_3': 'Item 3 - Legal Proceedings',
        'section_4': 'Item 4 - Mine Safety Disclosures',
        'section_5': 'Item 5 - Market Information',
        'section_6': 'Item 6 - Selected Financial Data',
        'section_7': 'Item 7 - Management Discussion & Analysis',
        'section_7A': 'Item 7A - Market Risk Disclosures',
        'section_8': 'Item 8 - Financial Statements',
        'section_9': 'Item 9 - Auditor Changes',
        'section_9A': 'Item 9A - Controls and Procedures',
        'section_9B': 'Item 9B - Other Information',
        'section_10': 'Item 10 - Directors & Officers',
        'section_11': 'Item 11 - Executive Compensation',
        'section_12': 'Item 12 - Security Ownership',
        'section_13': 'Item 13 - Related Transactions',
        'section_14': 'Item 14 - Accountant Fees',
        'section_15': 'Item 15 - Exhibits & Schedules'
    }
    
    def __init__(self, max_features: int = 3000):
        """
        Initialize the document classifier.
        
        Args:
            max_features: Maximum number of TF-IDF features (default 3000)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        self.model: Optional[keras.Model] = None
        self.label_encoder = LabelEncoder()
        self.classes = ['10-K', '10-Q', '8-K']
        self.max_features = max_features
        self._is_fitted = False
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess document text.
        
        Args:
            text: Raw document text
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags if present
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\%\$]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncate very long documents (keep first 50000 chars)
        if len(text) > 50000:
            text = text[:50000]
        
        return text
    
    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build the Keras neural network architecture.
        
        Args:
            input_dim: Dimension of input features (TF-IDF vector size)
            
        Returns:
            Compiled Keras Sequential model
        """
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=(input_dim,)),
            
            # First hidden layer with batch normalization
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),
            
            # Second hidden layer
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            # Third hidden layer
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            
            # Output layer
            keras.layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 15,
        validation_split: float = 0.2,
        batch_size: int = 32,
        early_stopping: bool = True
    ) -> keras.callbacks.History:
        """
        Train the document classifier.
        
        Args:
            texts: List of document texts
            labels: List of document type labels
            epochs: Number of training epochs
            validation_split: Fraction of data for validation
            batch_size: Training batch size
            early_stopping: Whether to use early stopping
            
        Returns:
            Keras History object with training metrics
        """
        print(f"Preprocessing {len(texts)} documents...")
        processed_texts = [self._preprocess_text(t) for t in texts]
        
        print("Fitting TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(processed_texts).toarray()
        
        print("Encoding labels...")
        y = self.label_encoder.fit_transform(labels)
        self.classes = list(self.label_encoder.classes_)
        
        print(f"Building model with input dimension: {X.shape[1]}")
        self.model = self._build_model(X.shape[1])
        
        # Print model summary
        self.model.summary()
        
        # Setup callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Add learning rate reduction
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=0.0001,
                verbose=1
            )
        )
        
        print("Training model...")
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self._is_fitted = True
        print("Training complete!")
        
        return history
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Classify a single document.
        
        Args:
            text: Document text to classify
            
        Returns:
            Tuple of (predicted_class, confidence_score)
        """
        if not self._is_fitted and self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        processed = self._preprocess_text(text)
        X = self.vectorizer.transform([processed]).toarray()
        
        probs = self.model.predict(X, verbose=0)[0]
        predicted_idx = np.argmax(probs)
        predicted_class = self.classes[predicted_idx]
        confidence = float(probs[predicted_idx])
        
        return predicted_class, confidence
    
    def get_readable_label(self, section_code: str) -> str:
        """
        Get human-readable label for a section code.
        
        Args:
            section_code: Section code like 'section_1', 'section_7A'
            
        Returns:
            Human-readable label like 'Item 1 - Business Overview'
        """
        return self.SECTION_LABELS.get(section_code, section_code)
    
    def predict_with_label(self, text: str) -> Tuple[str, str, float]:
        """
        Classify a document and return both code and readable label.
        
        Args:
            text: Document text to classify
            
        Returns:
            Tuple of (section_code, readable_label, confidence_score)
        """
        section_code, confidence = self.predict(text)
        readable_label = self.get_readable_label(section_code)
        return section_code, readable_label, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Classify multiple documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of (predicted_class, confidence_score) tuples
        """
        if not self._is_fitted and self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        processed = [self._preprocess_text(t) for t in texts]
        X = self.vectorizer.transform(processed).toarray()
        
        probs = self.model.predict(X, verbose=0)
        
        results = []
        for prob in probs:
            predicted_idx = np.argmax(prob)
            results.append((self.classes[predicted_idx], float(prob[predicted_idx])))
        
        return results
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get probability scores for all classes.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping class names to probabilities
        """
        if not self._is_fitted and self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        processed = self._preprocess_text(text)
        X = self.vectorizer.transform([processed]).toarray()
        probs = self.model.predict(X, verbose=0)[0]
        
        return {cls: float(prob) for cls, prob in zip(self.classes, probs)}
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            texts: List of test documents
            labels: List of true labels
            
        Returns:
            Dictionary with loss and accuracy metrics
        """
        processed = [self._preprocess_text(t) for t in texts]
        X = self.vectorizer.transform(processed).toarray()
        y = self.label_encoder.transform(labels)
        
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        return {'loss': loss, 'accuracy': accuracy}
    
    def save(self, path: str) -> None:
        """
        Save model and vectorizer to disk.
        
        Args:
            path: Directory path to save model files
        """
        os.makedirs(path, exist_ok=True)
        
        # Save Keras model
        self.model.save(os.path.join(path, "classifier_model.keras"))
        
        # Save vectorizer and label encoder
        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.joblib"))
        joblib.dump(self.label_encoder, os.path.join(path, "label_encoder.joblib"))
        joblib.dump(self.classes, os.path.join(path, "classes.joblib"))
        
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load saved model and vectorizer.
        
        Args:
            path: Directory path containing saved model files
        """
        # Load Keras model
        self.model = keras.models.load_model(
            os.path.join(path, "classifier_model.keras")
        )
        
        # Load vectorizer and label encoder
        self.vectorizer = joblib.load(os.path.join(path, "vectorizer.joblib"))
        self.label_encoder = joblib.load(os.path.join(path, "label_encoder.joblib"))
        self.classes = joblib.load(os.path.join(path, "classes.joblib"))
        
        self._is_fitted = True
        print("  ✓ Classifier loaded")


if __name__ == "__main__":
    # Quick test
    classifier = DocumentClassifier()
    
    # Sample test data
    sample_texts = [
        "Annual report pursuant to section 13 of the securities exchange act. " * 50,
        "Quarterly report pursuant to section 13 of the securities exchange act. " * 50,
        "Current report pursuant to section 13 of the securities exchange act. " * 50,
    ] * 10
    
    sample_labels = ['10-K', '10-Q', '8-K'] * 10
    
    print("Training sample classifier...")
    history = classifier.train(sample_texts, sample_labels, epochs=5)
    
    print("\nTesting prediction...")
    pred_class, confidence = classifier.predict(sample_texts[0])
    print(f"Predicted: {pred_class} with confidence {confidence:.2%}")
