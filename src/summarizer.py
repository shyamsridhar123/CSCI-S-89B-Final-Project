"""
Extractive Summarizer Module
Fast extractive summarization using TF-IDF sentence scoring.

No GPU required - runs quickly on CPU. Uses NLTK for sentence
tokenization and scikit-learn for TF-IDF scoring.

Author: Shyam Sridhar
"""

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Optional, Tuple
import re


class ExtractiveSummarizer:
    """
    Fast extractive summarization using TF-IDF sentence scoring.
    
    Extracts the most important sentences from a document based on
    TF-IDF scores and optional position weighting.
    
    No GPU required - runs quickly on CPU.
    """
    
    def __init__(self):
        """Initialize the summarizer and download required NLTK data."""
        # Download required NLTK data
        for resource in ['punkt', 'punkt_tab', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
        
        self.stopwords = set()
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words('english'))
        except:
            pass
        
        print("  ✓ Summarizer ready")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean text for summarization.
        
        Args:
            text: Raw document text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very long strings of numbers (likely tables)
        text = re.sub(r'[\d\s,\.]{50,}', ' ', text)
        
        return text.strip()
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Document text
            
        Returns:
            List of sentences
        """
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback to simple splitting
            sentences = re.split(r'[.!?]+', text)
        
        # Filter out very short or very long sentences
        sentences = [
            s.strip() for s in sentences
            if 20 < len(s.strip()) < 1000
        ]
        
        return sentences
    
    def _calculate_sentence_scores(
        self,
        sentences: List[str],
        position_weight: bool = True
    ) -> np.ndarray:
        """
        Calculate importance scores for each sentence.
        
        Args:
            sentences: List of sentences
            position_weight: Whether to weight by position
            
        Returns:
            Array of sentence scores
        """
        if len(sentences) == 0:
            return np.array([])
        
        if len(sentences) == 1:
            return np.array([1.0])
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # All sentences are stop words or empty
            return np.ones(len(sentences)) / len(sentences)
        
        # Calculate sentence scores as sum of TF-IDF values
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Normalize
        if sentence_scores.max() > 0:
            sentence_scores = sentence_scores / sentence_scores.max()
        
        # Apply position weighting (first and last sentences are often important)
        if position_weight:
            position_weights = np.ones(len(sentences))
            
            # Boost first 3 sentences
            for i in range(min(3, len(sentences))):
                position_weights[i] *= 1.5 - (i * 0.1)
            
            # Slight boost to last 2 sentences
            for i in range(max(0, len(sentences) - 2), len(sentences)):
                position_weights[i] *= 1.2
            
            sentence_scores = sentence_scores * position_weights
        
        return sentence_scores
    
    def summarize(
        self,
        text: str,
        num_sentences: int = 5,
        position_weight: bool = True,
        min_sentence_length: int = 30
    ) -> str:
        """
        Extract top N most important sentences as summary.
        
        Args:
            text: Document text to summarize
            num_sentences: Number of sentences in summary
            position_weight: Whether to weight by position
            min_sentence_length: Minimum characters per sentence
            
        Returns:
            Summary text (concatenated sentences)
        """
        # Preprocess
        text = self._preprocess_text(text)
        
        # Tokenize into sentences
        sentences = self._tokenize_sentences(text)
        
        # Handle edge cases
        if len(sentences) == 0:
            return "Unable to generate summary: no valid sentences found."
        
        if len(sentences) <= num_sentences:
            return ' '.join(sentences)
        
        # Calculate sentence scores
        scores = self._calculate_sentence_scores(sentences, position_weight)
        
        # Get top N sentence indices (preserve original order)
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices = sorted(top_indices)  # Sort to preserve order
        
        # Build summary
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def summarize_with_scores(
        self,
        text: str,
        num_sentences: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get summary sentences with their importance scores.
        
        Args:
            text: Document text
            num_sentences: Number of sentences to return
            
        Returns:
            List of (sentence, score) tuples
        """
        text = self._preprocess_text(text)
        sentences = self._tokenize_sentences(text)
        
        if len(sentences) == 0:
            return []
        
        if len(sentences) <= num_sentences:
            return [(s, 1.0) for s in sentences]
        
        scores = self._calculate_sentence_scores(sentences)
        
        # Get top N
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices = sorted(top_indices)
        
        return [(sentences[i], float(scores[i])) for i in top_indices]
    
    def get_key_sentences(
        self,
        text: str,
        threshold: float = 0.5
    ) -> List[str]:
        """
        Get all sentences above importance threshold.
        
        Args:
            text: Document text
            threshold: Minimum importance score (0-1)
            
        Returns:
            List of important sentences
        """
        text = self._preprocess_text(text)
        sentences = self._tokenize_sentences(text)
        
        if len(sentences) == 0:
            return []
        
        scores = self._calculate_sentence_scores(sentences)
        
        # Filter by threshold
        important = [
            sentences[i] for i in range(len(sentences))
            if scores[i] >= threshold
        ]
        
        return important if important else [sentences[np.argmax(scores)]]
    
    def summarize_sections(
        self,
        text: str,
        section_headers: Optional[List[str]] = None,
        sentences_per_section: int = 2
    ) -> str:
        """
        Summarize a document by sections.
        
        Args:
            text: Document text
            section_headers: List of section headers to look for
            sentences_per_section: Sentences to extract per section
            
        Returns:
            Section-wise summary
        """
        if section_headers is None:
            section_headers = [
                'Executive Summary', 'Overview', 'Financial Highlights',
                'Results of Operations', 'Risk Factors', 'Outlook'
            ]
        
        # Try to find sections
        sections = {}
        current_section = 'General'
        current_text = []
        
        for line in text.split('\n'):
            # Check if line is a section header
            is_header = False
            for header in section_headers:
                if header.lower() in line.lower():
                    if current_text:
                        sections[current_section] = ' '.join(current_text)
                    current_section = header
                    current_text = []
                    is_header = True
                    break
            
            if not is_header:
                current_text.append(line)
        
        # Add last section
        if current_text:
            sections[current_section] = ' '.join(current_text)
        
        # Summarize each section
        summaries = []
        for section_name, section_text in sections.items():
            if len(section_text.strip()) > 100:
                summary = self.summarize(section_text, num_sentences=sentences_per_section)
                summaries.append(f"**{section_name}**: {summary}")
        
        if not summaries:
            return self.summarize(text)
        
        return '\n\n'.join(summaries)


if __name__ == "__main__":
    # Quick test
    summarizer = ExtractiveSummarizer()
    
    test_text = """
    Apple Inc. today announced financial results for its fiscal 2024 fourth quarter 
    ended September 28, 2024. The Company posted quarterly revenue of $94.9 billion, 
    up 6 percent year over year, and quarterly earnings per diluted share of $1.64, 
    up 12 percent year over year.
    
    iPhone revenue was $46.2 billion, up 6 percent year over year. Mac revenue was 
    $7.7 billion, down 2 percent year over year. iPad revenue was $7.0 billion, up 
    8 percent year over year. Wearables, Home and Accessories revenue was $9.0 billion, 
    down 3 percent year over year. Services revenue was $25.0 billion, up 12 percent 
    year over year.
    
    The Company's board of directors has declared a cash dividend of $0.25 per share 
    of the Company's common stock. The dividend is payable on November 14, 2024, to 
    shareholders of record as of the close of business on November 11, 2024.
    
    Apple's cash position remains strong with $162.3 billion in cash and marketable 
    securities at the end of the quarter. During the quarter, the Company returned 
    over $29 billion to shareholders through dividends and share repurchases.
    
    Looking ahead, Apple expects continued growth in its Services segment and remains 
    optimistic about the upcoming holiday season. The company is investing heavily in 
    artificial intelligence capabilities across its product lineup.
    """
    
    print("\n" + "="*60)
    print("Testing Extractive Summarizer")
    print("="*60)
    
    summary = summarizer.summarize(test_text, num_sentences=3)
    print(f"\nOriginal text: {len(test_text)} characters")
    print(f"Summary: {len(summary)} characters")
    print(f"\nSummary:\n{summary}")
    
    print("\n" + "-"*40)
    print("Summary with scores:")
    scored = summarizer.summarize_with_scores(test_text, num_sentences=3)
    for sentence, score in scored:
        print(f"\n[Score: {score:.3f}] {sentence[:100]}...")
