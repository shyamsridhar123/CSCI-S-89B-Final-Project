"""
Sentiment Analyzer Module
Financial sentiment analysis using FinBERT.

Uses the ProsusAI/finbert model fine-tuned on financial text
for accurate financial sentiment classification.

Author: Shyam Sridhar
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Optional
import warnings

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", message=".*tokenizer.*")


class SentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT.
    
    FinBERT is a BERT model fine-tuned on financial text that understands
    domain-specific language and context (e.g., "bearish", "bullish",
    "revenue miss", "guidance raised").
    
    Returns sentiment labels: positive, negative, or neutral with confidence scores.
    
    Attributes:
        pipeline: HuggingFace sentiment-analysis pipeline
        max_length: Maximum token length for input text
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier (default: ProsusAI/finbert)
        """
        self.model_name = model_name
        self.max_length = 512  # FinBERT token limit
        
        # Check if model is cached
        import os
        cache_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
        is_cached = os.path.exists(cache_path)
        
        if is_cached:
            print(f"Loading {model_name} from cache...")
        else:
            print(f"Downloading {model_name} (first run only, ~500MB)...")
        
        # Load model and tokenizer (uses cache if available)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=self.max_length
        )
        
        print("  ✓ FinBERT ready")
    
    def _truncate_text(self, text: str, max_words: int = 400) -> str:
        """
        Truncate text to approximate token limit.
        
        Args:
            text: Input text
            max_words: Maximum number of words (rough token approximation)
            
        Returns:
            Truncated text
        """
        words = text.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words])
        return text
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Document or text to analyze
            
        Returns:
            Dictionary containing:
                - label: Sentiment label (positive/negative/neutral)
                - score: Confidence score (0-1)
                - interpretation: Human-readable interpretation
        """
        if not text or len(text.strip()) < 10:
            return {
                'label': 'neutral',
                'score': 0.0,
                'interpretation': 'Text too short for analysis'
            }
        
        # Truncate to max length
        truncated = self._truncate_text(text)
        
        try:
            result = self.pipeline(truncated)[0]
            
            return {
                'label': result['label'].lower(),
                'score': round(result['score'], 4),
                'interpretation': self._interpret(result['label'], result['score'])
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'interpretation': f'Analysis failed: {str(e)}'
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of documents/texts to analyze
            
        Returns:
            List of sentiment result dictionaries
        """
        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results
    
    def analyze_sections(self, text: str, section_size: int = 500) -> Dict:
        """
        Analyze sentiment across document sections and aggregate.
        
        Useful for long documents where overall sentiment may vary.
        
        Args:
            text: Full document text
            section_size: Number of words per section
            
        Returns:
            Dictionary with overall and section-level sentiment
        """
        words = text.split()
        sections = []
        
        # Split into sections
        for i in range(0, len(words), section_size):
            section_text = ' '.join(words[i:i + section_size])
            if len(section_text.strip()) > 50:
                sections.append(section_text)
        
        if not sections:
            return self.analyze(text)
        
        # Analyze each section
        section_results = []
        positive_scores = []
        negative_scores = []
        neutral_scores = []
        
        for section in sections:
            result = self.analyze(section)
            section_results.append(result)
            
            if result['label'] == 'positive':
                positive_scores.append(result['score'])
            elif result['label'] == 'negative':
                negative_scores.append(result['score'])
            else:
                neutral_scores.append(result['score'])
        
        # Aggregate results
        total_sections = len(sections)
        avg_positive = sum(positive_scores) / total_sections if positive_scores else 0
        avg_negative = sum(negative_scores) / total_sections if negative_scores else 0
        avg_neutral = sum(neutral_scores) / total_sections if neutral_scores else 0
        
        # Determine overall sentiment
        scores = {
            'positive': len(positive_scores) / total_sections,
            'negative': len(negative_scores) / total_sections,
            'neutral': len(neutral_scores) / total_sections
        }
        overall_label = max(scores, key=scores.get)
        
        return {
            'overall': {
                'label': overall_label,
                'score': scores[overall_label],
                'interpretation': self._interpret(overall_label, scores[overall_label])
            },
            'distribution': {
                'positive': len(positive_scores),
                'negative': len(negative_scores),
                'neutral': len(neutral_scores)
            },
            'section_count': total_sections,
            'sections': section_results
        }
    
    def _interpret(self, label: str, score: float) -> str:
        """
        Generate human-readable interpretation.
        
        Args:
            label: Sentiment label
            score: Confidence score
            
        Returns:
            Interpretation string
        """
        label = label.lower()
        
        if score > 0.85:
            strength = "strongly"
        elif score > 0.7:
            strength = "moderately"
        elif score > 0.5:
            strength = "slightly"
        else:
            strength = "marginally"
        
        return f"Document is {strength} {label} (confidence: {score:.1%})"
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get a single sentiment score from -1 (negative) to +1 (positive).
        
        Args:
            text: Text to analyze
            
        Returns:
            Float score from -1 to +1
        """
        result = self.analyze(text)
        
        if result['label'] == 'positive':
            return result['score']
        elif result['label'] == 'negative':
            return -result['score']
        else:
            return 0.0


if __name__ == "__main__":
    # Quick test
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "The company reported strong quarterly earnings, beating analyst expectations by 15%.",
        "Revenue declined 20% year-over-year due to supply chain disruptions and weak demand.",
        "The board of directors announced a regular quarterly dividend of $0.25 per share."
    ]
    
    print("\n" + "="*60)
    print("Testing Sentiment Analyzer")
    print("="*60)
    
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\nText: {text[:80]}...")
        print(f"Sentiment: {result['label'].upper()}")
        print(f"Confidence: {result['score']:.2%}")
        print(f"Interpretation: {result['interpretation']}")
