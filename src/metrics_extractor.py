"""
Financial Metrics Extraction Module
Extracts and highlights key financial metrics from SEC filings.

Supports FinBERT-enhanced forward-looking detection for improved
accuracy and confidence scoring.

Author: Shyam Sridhar
CSCI S-89B Final Project
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class FinancialMetric:
    """Represents an extracted financial metric."""
    metric_type: str
    value: str
    raw_match: str
    context: str
    position: int
    unit: Optional[str] = None
    change_direction: Optional[str] = None  # 'increase', 'decrease', None
    change_percent: Optional[float] = None


class MetricsExtractor:
    """
    Extracts key financial metrics from documents.
    
    Identifies revenue figures, percentages, year-over-year changes,
    margins, and forward-looking guidance.
    """
    
    # Metric patterns with named groups
    METRIC_PATTERNS = {
        'revenue': r'(?:revenue|sales|net\s+sales)\s+(?:of|was|were|totaled|reached)?\s*\$?([\d,.]+)\s*(million|billion|M|B|thousand|K)?',
        
        'revenue_change': r'(?:revenue|sales)\s+(?:increased|decreased|grew|declined|rose|fell)\s+(?:by\s+)?\$?([\d,.]+)\s*(million|billion|M|B)?(?:\s+or\s+([\d.]+)\s*%)?',
        
        'net_income': r'(?:net\s+income|earnings|profit)\s+(?:of|was|were|totaled)?\s*\$?([\d,.]+)\s*(million|billion|M|B)?',
        
        'eps': r'(?:earnings\s+per\s+share|EPS)\s+(?:of|was|were)?\s*\$?([\d.]+)',
        
        'margin': r'(?:gross|operating|net|profit)\s+margin\s+(?:of|was|were|is)?\s*([\d.]+)\s*%',
        
        'yoy_change': r'([\d.]+)\s*%\s+(?:increase|decrease|growth|decline|higher|lower)\s+(?:year-over-year|YoY|y\/y|compared\s+to\s+(?:the\s+)?(?:prior|previous|last)\s+year)',
        
        'qoq_change': r'([\d.]+)\s*%\s+(?:increase|decrease|growth|decline)\s+(?:quarter-over-quarter|QoQ|q\/q|sequentially|compared\s+to\s+(?:the\s+)?(?:prior|previous|last)\s+quarter)',
        
        'guidance': r'(?:expect|anticipate|project|forecast|guide|outlook)\s+.*?\$?([\d,.]+)\s*(million|billion|M|B)?',
        
        'cash_flow': r'(?:cash\s+flow|operating\s+cash|free\s+cash\s+flow)\s+(?:of|was|were|totaled)?\s*\$?([\d,.]+)\s*(million|billion|M|B)?',
        
        'debt': r'(?:total\s+debt|long-term\s+debt|debt)\s+(?:of|was|were|totaled)?\s*\$?([\d,.]+)\s*(million|billion|M|B)?',
        
        'assets': r'(?:total\s+assets)\s+(?:of|was|were|totaled)?\s*\$?([\d,.]+)\s*(million|billion|M|B)?',
        
        'dividend': r'(?:dividend|quarterly\s+dividend)\s+(?:of|was|at)?\s*\$?([\d.]+)\s*(?:per\s+share)?',
        
        'share_count': r'([\d,.]+)\s*(million|billion|M|B)?\s+(?:shares|common\s+shares)\s+(?:outstanding|issued)',
        
        'headcount': r'(?:employees?|workforce|headcount|personnel)\s+(?:of|was|were|totaled|approximately)?\s*([\d,]+)',
    }
    
    # Money value patterns
    MONEY_PATTERN = re.compile(
        r'\$\s*([\d,.]+)\s*(million|billion|trillion|M|B|K|thousand)?',
        re.IGNORECASE
    )
    
    # Percentage pattern
    PERCENTAGE_PATTERN = re.compile(
        r'([\d.]+)\s*%|(?:percent(?:age)?)\s+(?:of\s+)?([\d.]+)',
        re.IGNORECASE
    )
    
    # Metric type labels for display
    METRIC_LABELS = {
        'revenue': '💵 Revenue',
        'revenue_change': '📊 Revenue Change',
        'net_income': '💰 Net Income',
        'eps': '📈 Earnings Per Share',
        'margin': '📉 Margin',
        'yoy_change': '📆 Year-over-Year Change',
        'qoq_change': '📅 Quarter-over-Quarter Change',
        'guidance': '🔮 Forward Guidance',
        'cash_flow': '💸 Cash Flow',
        'debt': '📋 Debt',
        'assets': '🏦 Assets',
        'dividend': '💎 Dividend',
        'share_count': '📊 Shares Outstanding',
        'headcount': '👥 Employee Count'
    }
    
    def __init__(self):
        """Initialize the metrics extractor with compiled patterns."""
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.METRIC_PATTERNS.items()
        }
        print("  ✓ Metrics extractor ready")
    
    def extract(self, text: str) -> List[FinancialMetric]:
        """
        Extract all financial metrics from text.
        
        Args:
            text: Document text to analyze
            
        Returns:
            List of FinancialMetric objects
        """
        metrics = []
        
        for metric_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                # Get context
                start = max(0, match.start() - 75)
                end = min(len(text), match.end() + 75)
                context = text[start:end].strip()
                
                # Parse value
                value = match.group(1) if match.groups() else match.group()
                unit = match.group(2) if len(match.groups()) > 1 else None
                
                # Detect change direction
                change_direction = None
                change_percent = None
                
                if 'increase' in match.group().lower() or 'grew' in match.group().lower() or 'rose' in match.group().lower():
                    change_direction = 'increase'
                elif 'decrease' in match.group().lower() or 'decline' in match.group().lower() or 'fell' in match.group().lower():
                    change_direction = 'decrease'
                
                # Try to extract percentage if in change context
                if len(match.groups()) > 2 and match.group(3):
                    try:
                        change_percent = float(match.group(3))
                    except (ValueError, TypeError):
                        pass
                
                metrics.append(FinancialMetric(
                    metric_type=metric_type,
                    value=value,
                    raw_match=match.group(),
                    context=context,
                    position=match.start(),
                    unit=unit,
                    change_direction=change_direction,
                    change_percent=change_percent
                ))
        
        # Sort by position in document
        metrics.sort(key=lambda m: m.position)
        
        return metrics
    
    def extract_money_values(self, text: str) -> List[Dict]:
        """
        Extract all monetary values from text.
        
        Args:
            text: Document text
            
        Returns:
            List of money value dictionaries
        """
        values = []
        for match in self.MONEY_PATTERN.finditer(text):
            amount = match.group(1).replace(',', '')
            unit = match.group(2) or ''
            
            # Normalize value
            try:
                numeric = float(amount)
                unit_lower = unit.lower() if unit else ''
                
                if unit_lower in ['billion', 'b']:
                    normalized = numeric * 1e9
                elif unit_lower in ['million', 'm']:
                    normalized = numeric * 1e6
                elif unit_lower in ['thousand', 'k']:
                    normalized = numeric * 1e3
                else:
                    normalized = numeric
                
                # Get context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                values.append({
                    'raw': match.group(),
                    'amount': numeric,
                    'unit': unit,
                    'normalized': normalized,
                    'context': context,
                    'position': match.start()
                })
            except ValueError:
                continue
        
        return values
    
    def get_metrics_summary(self, metrics: List[FinancialMetric]) -> Dict:
        """
        Generate a summary of extracted metrics.
        
        Args:
            metrics: List of extracted metrics
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_count': len(metrics),
            'by_type': {},
            'key_metrics': [],
            'changes': []
        }
        
        # Group by type
        type_counts = {}
        for m in metrics:
            if m.metric_type not in type_counts:
                type_counts[m.metric_type] = []
            type_counts[m.metric_type].append(m)
        
        for metric_type, items in type_counts.items():
            summary['by_type'][metric_type] = {
                'label': self.METRIC_LABELS.get(metric_type, metric_type),
                'count': len(items),
                'examples': [i.raw_match for i in items[:3]]
            }
        
        # Identify key metrics (first occurrence of each type)
        seen_types = set()
        for m in metrics:
            if m.metric_type not in seen_types:
                seen_types.add(m.metric_type)
                summary['key_metrics'].append({
                    'type': m.metric_type,
                    'label': self.METRIC_LABELS.get(m.metric_type, m.metric_type),
                    'value': m.value,
                    'unit': m.unit,
                    'context': m.context
                })
        
        # Extract changes
        for m in metrics:
            if m.change_direction:
                summary['changes'].append({
                    'type': m.metric_type,
                    'direction': m.change_direction,
                    'percent': m.change_percent,
                    'raw': m.raw_match
                })
        
        return summary
    
    def format_for_display(self, metrics: List[FinancialMetric]) -> str:
        """
        Format metrics for markdown display.
        
        Args:
            metrics: List of extracted metrics
            
        Returns:
            Markdown-formatted string
        """
        if not metrics:
            return "ℹ️ **No specific financial metrics detected.**"
        
        summary = self.get_metrics_summary(metrics)
        
        output = []
        output.append("## 📊 Key Financial Metrics\n\n")
        output.append(f"**Total Metrics Found**: {summary['total_count']}\n\n")
        
        # Key metrics
        if summary['key_metrics']:
            output.append("### Highlighted Metrics\n\n")
            for km in summary['key_metrics'][:8]:
                label = km['label']
                value = km['value']
                unit = km['unit'] or ''
                output.append(f"- {label}: **${value} {unit}**\n")
        
        # Changes
        if summary['changes']:
            output.append("\n### Performance Changes\n\n")
            for change in summary['changes'][:5]:
                direction = '📈' if change['direction'] == 'increase' else '📉'
                pct = f" ({change['percent']}%)" if change['percent'] else ''
                output.append(f"- {direction} {change['raw']}{pct}\n")
        
        # Metric type breakdown
        output.append("\n### Metrics by Category\n\n")
        for metric_type, info in summary['by_type'].items():
            output.append(f"- {info['label']}: {info['count']} mentions\n")
        
        return ''.join(output)


class ForwardLookingDetector:
    """
    Detects forward-looking statements in financial documents.
    
    Identifies predictive language and categorizes by confidence level.
    Optionally uses FinBERT for enhanced sentiment-based scoring.
    """
    
    # Forward-looking indicators by confidence
    HIGH_CONFIDENCE = [
        'will', 'expect', 'plan to', 'intend to', 'committed to',
        'on track to', 'scheduled to', 'set to'
    ]
    
    MEDIUM_CONFIDENCE = [
        'believe', 'anticipate', 'project', 'forecast', 'estimate',
        'aim to', 'seek to', 'target'
    ]
    
    LOW_CONFIDENCE = [
        'may', 'could', 'might', 'possible', 'potentially',
        'would', 'should', 'hope to'
    ]
    
    # Regex patterns for more accurate matching
    HIGH_CONFIDENCE_PATTERNS = [
        r'\bwill\b', r'\bexpect(?:s|ed|ing)?\b', r'\bplan(?:s|ned|ning)?\s+to\b',
        r'\bintend(?:s|ed|ing)?\s+to\b', r'\bcommitted\s+to\b',
        r'\bon\s+track\s+to\b', r'\bscheduled\s+to\b', r'\bset\s+to\b',
        r'\bwill\s+continue\b', r'\bgoing\s+forward\b'
    ]
    
    MEDIUM_CONFIDENCE_PATTERNS = [
        r'\bbelieve(?:s|d)?\b', r'\banticipate(?:s|d)?\b',
        r'\bproject(?:s|ed|ing)?\b', r'\bforecast(?:s|ed|ing)?\b',
        r'\bestimate(?:s|d)?\b', r'\baim(?:s|ed|ing)?\s+to\b',
        r'\bseek(?:s|ing)?\s+to\b', r'\btarget(?:s|ed|ing)?\b',
        r'\bpositioned\s+to\b', r'\bprepared\s+to\b'
    ]
    
    LOW_CONFIDENCE_PATTERNS = [
        r'\bmay\b', r'\bcould\b', r'\bmight\b', r'\bpossible\b',
        r'\bpotentially\b', r'\bwould\b', r'\bshould\b',
        r'\bhope(?:s|d)?\s+to\b', r'\blikely\b'
    ]
    
    def __init__(self, finbert_pipeline: Optional[Any] = None):
        """
        Initialize detector.
        
        Args:
            finbert_pipeline: Optional FinBERT pipeline for enhanced scoring.
                             If provided, uses FinBERT for sentiment analysis.
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        self.finbert = finbert_pipeline
        self._use_finbert = finbert_pipeline is not None
        
        # Compile regex patterns for enhanced matching
        self._high_patterns = [re.compile(p, re.IGNORECASE) for p in self.HIGH_CONFIDENCE_PATTERNS]
        self._medium_patterns = [re.compile(p, re.IGNORECASE) for p in self.MEDIUM_CONFIDENCE_PATTERNS]
        self._low_patterns = [re.compile(p, re.IGNORECASE) for p in self.LOW_CONFIDENCE_PATTERNS]
        
        mode = "FinBERT-enhanced" if self._use_finbert else "keyword-based"
        print(f"  ✓ Forward-looking detector ready ({mode})")
    
    def _get_keyword_confidence(self, sentence: str) -> Tuple[Optional[str], Optional[str]]:
        """Check for forward-looking keywords using regex patterns."""
        for pattern in self._high_patterns:
            match = pattern.search(sentence)
            if match:
                return 'high', match.group()
        
        for pattern in self._medium_patterns:
            match = pattern.search(sentence)
            if match:
                return 'medium', match.group()
        
        for pattern in self._low_patterns:
            match = pattern.search(sentence)
            if match:
                return 'low', match.group()
        
        return None, None
    
    def _get_finbert_score(self, sentence: str) -> Dict:
        """Get FinBERT sentiment for the sentence."""
        if not self.finbert:
            return {'label': 'neutral', 'score': 0.5}
        
        try:
            result = self.finbert(sentence[:512])[0]
            return {
                'label': result['label'].lower(),
                'score': result['score']
            }
        except Exception:
            return {'label': 'neutral', 'score': 0.5}
    
    def detect(self, text: str) -> List[Dict]:
        """
        Detect forward-looking statements.
        
        Args:
            text: Document text
            
        Returns:
            List of forward-looking statements with confidence levels.
            When FinBERT is enabled, includes sentiment and combined scores.
        """
        import nltk
        sentences = nltk.sent_tokenize(text)
        
        forward_looking = []
        
        for sent in sentences:
            # Use regex patterns for more accurate matching
            keyword_conf, trigger = self._get_keyword_confidence(sent)
            
            if keyword_conf:
                result = {
                    'sentence': sent,
                    'confidence': keyword_conf,
                    'trigger': trigger
                }
                
                # Add FinBERT analysis if available
                if self._use_finbert:
                    finbert_result = self._get_finbert_score(sent)
                    result['finbert_sentiment'] = finbert_result['label']
                    result['finbert_score'] = finbert_result['score']
                    
                    # Calculate combined score
                    keyword_weight = {'high': 0.9, 'medium': 0.7, 'low': 0.5}[keyword_conf]
                    sentiment_modifier = {
                        'positive': 1.0,
                        'neutral': 0.9,
                        'negative': 0.8
                    }.get(finbert_result['label'], 0.9)
                    
                    result['combined_score'] = keyword_weight * sentiment_modifier * finbert_result['score']
                
                forward_looking.append(result)
        
        # Sort by combined score if FinBERT is used
        if self._use_finbert and forward_looking:
            forward_looking.sort(key=lambda x: -x.get('combined_score', 0))
        
        return forward_looking
    
    def get_summary(self, statements: List[Dict]) -> Dict:
        """Get summary of forward-looking statements."""
        if not statements:
            return {
                'total': 0,
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0,
                'examples': []
            }
        
        summary = {
            'total': len(statements),
            'high_confidence': len([s for s in statements if s['confidence'] == 'high']),
            'medium_confidence': len([s for s in statements if s['confidence'] == 'medium']),
            'low_confidence': len([s for s in statements if s['confidence'] == 'low']),
            'examples': statements[:5]
        }
        
        # Add FinBERT-specific stats if available
        if self._use_finbert and statements:
            summary['avg_combined_score'] = sum(
                s.get('combined_score', 0) for s in statements
            ) / len(statements)
            summary['sentiment_breakdown'] = {
                'positive': len([s for s in statements if s.get('finbert_sentiment') == 'positive']),
                'neutral': len([s for s in statements if s.get('finbert_sentiment') == 'neutral']),
                'negative': len([s for s in statements if s.get('finbert_sentiment') == 'negative'])
            }
        
        return summary
    
    def format_for_display(self, statements: List[Dict]) -> str:
        """Format statements for display."""
        if not statements:
            return "ℹ️ **No forward-looking statements detected.**"
        
        summary = self.get_summary(statements)
        
        output = []
        output.append("## 🔮 Forward-Looking Statements\n\n")
        output.append(f"**Total Found**: {summary['total']}\n\n")
        output.append(f"- 🟢 High Confidence: {summary['high_confidence']}\n")
        output.append(f"- 🟡 Medium Confidence: {summary['medium_confidence']}\n")
        output.append(f"- 🔴 Low Confidence: {summary['low_confidence']}\n\n")
        
        # Add FinBERT stats if available
        if 'sentiment_breakdown' in summary:
            sb = summary['sentiment_breakdown']
            output.append(f"**FinBERT Sentiment Analysis**:\n")
            output.append(f"- 📈 Positive outlook: {sb['positive']}\n")
            output.append(f"- ➖ Neutral outlook: {sb['neutral']}\n")
            output.append(f"- 📉 Negative outlook: {sb['negative']}\n\n")
        
        if summary['examples']:
            output.append("### Examples\n\n")
            for stmt in summary['examples'][:5]:
                conf = stmt['confidence']
                icon = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}[conf]
                # Truncate long sentences
                sent = stmt['sentence'][:200] + '...' if len(stmt['sentence']) > 200 else stmt['sentence']
                
                # Add sentiment if available
                if 'finbert_sentiment' in stmt:
                    sent_icon = {'positive': '📈', 'neutral': '➖', 'negative': '📉'}[stmt['finbert_sentiment']]
                    output.append(f"{icon}{sent_icon} *\"{sent}\"*\n\n")
                else:
                    output.append(f"{icon} *\"{sent}\"*\n\n")
        
        return ''.join(output)


if __name__ == "__main__":
    # Test the extractors
    extractor = MetricsExtractor()
    fwd_detector = ForwardLookingDetector()
    
    sample_text = """
    Our revenue increased by $166.4 million or 11.2% over the prior year period.
    Net income was $24.7 billion with earnings per share of $3.30.
    Gross margin was 42.5% compared to 41.2% in the prior year.
    We expect revenue to grow 15% year-over-year in the coming quarter.
    The Company may face headwinds from supply chain disruptions.
    Total debt was $85 billion and total assets were $320 billion.
    We declared a dividend of $0.83 per share.
    """
    
    metrics = extractor.extract(sample_text)
    print(extractor.format_for_display(metrics))
    print("\n" + "="*50 + "\n")
    
    statements = fwd_detector.detect(sample_text)
    print(fwd_detector.format_for_display(statements))
