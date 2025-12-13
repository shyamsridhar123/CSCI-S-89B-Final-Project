"""
Entity Extractor Module
Named Entity Recognition using spaCy + custom financial patterns.

Combines spaCy's pre-trained NER with custom regex patterns
for financial-specific entities like monetary values, percentages,
fiscal dates, and stock tickers.

Author: Shyam Sridhar
"""

import spacy
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional


class EntityExtractor:
    """
    Extract named entities using spaCy + custom financial patterns.
    
    Combines general NER (organizations, people, locations) with
    financial-specific patterns (money, percentages, fiscal dates).
    
    Attributes:
        nlp: spaCy language model
        patterns: Dictionary of regex patterns for financial entities
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the entity extractor.
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"  ✓ spaCy ({model_name}) ready")
        except OSError:
            print(f"  Downloading spaCy model: {model_name}...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
            print(f"  ✓ spaCy ({model_name}) ready")
        
        # Custom patterns for financial entities
        self.patterns = {
            'MONEY': [
                # $X million/billion/trillion
                r'\$[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion|M|B|K|mn|bn)?',
                # X million dollars
                r'[\d,]+(?:\.\d{1,2})?\s*(?:million|billion|trillion)\s*(?:dollars|USD)?',
            ],
            'PERCENTAGE': [
                r'[\d.]+\s*%',
                r'[\d.]+\s*percent',
                r'[\d.]+\s*basis\s*points?',
            ],
            'FISCAL_DATE': [
                r'(?:Q[1-4]|FY)\s*\'?\d{2,4}',
                r'(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?\d{4}',
                r'fiscal\s+(?:year\s+)?\d{4}',
                r'(?:FY|CY)\d{2,4}',
            ],
            'TICKER': [
                r'\b[A-Z]{1,5}\b(?=\s+(?:stock|shares|Inc|Corp|Company|Ltd))',
                r'(?:NASDAQ|NYSE|AMEX):\s*[A-Z]{1,5}',
            ],
            'FINANCIAL_METRIC': [
                r'(?:EPS|P/E|ROE|ROA|EBITDA|GAAP|non-GAAP)',
                r'(?:revenue|earnings|income|profit|loss|margin)\s+(?:of\s+)?\$?[\d,.]+',
            ],
            'RATIO': [
                r'\d+(?:\.\d+)?[xX]\s*(?:revenue|earnings|EBITDA)?',
                r'\d+:\d+\s*(?:ratio|split)?',
            ]
        }
        
        # Entity type descriptions for display
        self.entity_descriptions = {
            'ORG': 'Organization',
            'PERSON': 'Person',
            'GPE': 'Location/Country',
            'DATE': 'Date',
            'MONEY': 'Monetary Value',
            'PERCENTAGE': 'Percentage',
            'FISCAL_DATE': 'Fiscal Period',
            'TICKER': 'Stock Ticker',
            'FINANCIAL_METRIC': 'Financial Metric',
            'RATIO': 'Financial Ratio',
            'CARDINAL': 'Number',
            'ORDINAL': 'Ordinal',
            'QUANTITY': 'Quantity'
        }
    
    def extract(self, text: str, max_length: int = 100000) -> List[Dict]:
        """
        Extract all entities from text.
        
        Args:
            text: Document text to analyze
            max_length: Maximum text length to process (for performance)
            
        Returns:
            List of entity dictionaries with text, type, and source
        """
        entities = []
        
        # Truncate for performance if needed
        if len(text) > max_length:
            text = text[:max_length]
        
        # spaCy NER
        doc = self.nlp(text)
        spacy_types = {'ORG', 'PERSON', 'GPE', 'DATE', 'MONEY', 'CARDINAL', 'PERCENT'}
        
        for ent in doc.ents:
            if ent.label_ in spacy_types:
                # Map PERCENT to PERCENTAGE for consistency
                label = 'PERCENTAGE' if ent.label_ == 'PERCENT' else ent.label_
                entities.append({
                    'text': ent.text.strip(),
                    'type': label,
                    'source': 'spacy',
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        # Custom pattern matching
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group().strip(),
                        'type': entity_type,
                        'source': 'pattern',
                        'start': match.start(),
                        'end': match.end()
                    })
        
        # Deduplicate and clean
        entities = self._deduplicate(entities)
        entities = self._clean_entities(entities)
        
        return entities
    
    def _deduplicate(self, entities: List[Dict]) -> List[Dict]:
        """
        Remove duplicate entities.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []
        
        for ent in entities:
            # Normalize text for comparison
            key = (ent['text'].lower().strip(), ent['type'])
            if key not in seen and len(ent['text'].strip()) > 1:
                seen.add(key)
                unique.append(ent)
        
        return unique
    
    def _clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Clean and filter entities.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Cleaned list
        """
        cleaned = []
        
        for ent in entities:
            text = ent['text'].strip()
            
            # Skip very short entities (likely noise)
            if len(text) < 2:
                continue
            
            # Skip common false positives
            skip_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'}
            if text.lower() in skip_words:
                continue
            
            # Skip purely numeric for ORG type
            if ent['type'] == 'ORG' and text.replace(',', '').replace('.', '').isdigit():
                continue
            
            cleaned.append(ent)
        
        return cleaned
    
    def extract_by_type(self, text: str, entity_types: List[str]) -> List[Dict]:
        """
        Extract only specific entity types.
        
        Args:
            text: Document text
            entity_types: List of entity types to extract
            
        Returns:
            Filtered list of entities
        """
        all_entities = self.extract(text)
        return [e for e in all_entities if e['type'] in entity_types]
    
    def get_summary(self, entities: List[Dict]) -> Dict[str, int]:
        """
        Get entity type counts for visualization.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Dictionary mapping entity types to counts
        """
        type_counts = Counter(e['type'] for e in entities)
        return dict(type_counts)
    
    def get_top_entities(
        self,
        entities: List[Dict],
        n: int = 10,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get the most frequently mentioned entities.
        
        Args:
            entities: List of entity dictionaries
            n: Number of top entities to return
            entity_types: Filter by specific types (optional)
            
        Returns:
            List of top entities with counts
        """
        if entity_types:
            entities = [e for e in entities if e['type'] in entity_types]
        
        # Count occurrences
        text_counts = Counter(e['text'] for e in entities)
        
        # Get top N
        top_texts = text_counts.most_common(n)
        
        # Build result with entity info
        result = []
        for text, count in top_texts:
            # Find entity info
            ent = next((e for e in entities if e['text'] == text), None)
            if ent:
                result.append({
                    'text': text,
                    'type': ent['type'],
                    'count': count
                })
        
        return result
    
    def get_organizations(self, text: str) -> List[str]:
        """
        Extract organization names.
        
        Args:
            text: Document text
            
        Returns:
            List of organization names
        """
        entities = self.extract_by_type(text, ['ORG'])
        return list(set(e['text'] for e in entities))
    
    def get_monetary_values(self, text: str) -> List[str]:
        """
        Extract monetary values.
        
        Args:
            text: Document text
            
        Returns:
            List of monetary value strings
        """
        entities = self.extract_by_type(text, ['MONEY'])
        return list(set(e['text'] for e in entities))
    
    def format_entities_for_display(self, entities: List[Dict]) -> str:
        """
        Format entities as readable markdown.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Markdown-formatted string
        """
        if not entities:
            return "No entities found."
        
        # Group by type
        by_type: Dict[str, List[str]] = {}
        for ent in entities:
            ent_type = ent['type']
            if ent_type not in by_type:
                by_type[ent_type] = []
            if ent['text'] not in by_type[ent_type]:
                by_type[ent_type].append(ent['text'])
        
        # Format output
        lines = ["### Extracted Entities\n"]
        for ent_type, texts in sorted(by_type.items()):
            type_name = self.entity_descriptions.get(ent_type, ent_type)
            lines.append(f"**{type_name}** ({len(texts)})")
            for text in texts[:10]:  # Limit to 10 per type
                lines.append(f"  - {text}")
            if len(texts) > 10:
                lines.append(f"  - ... and {len(texts) - 10} more")
            lines.append("")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    # Quick test
    extractor = EntityExtractor()
    
    test_text = """
    Apple Inc. (NASDAQ: AAPL) reported Q4 2024 earnings of $1.46 per share, 
    beating analyst expectations by 5%. Revenue grew 8% year-over-year to 
    $89.5 billion. CEO Tim Cook announced a $100 billion share buyback program.
    The company's P/E ratio stands at 28.5x, with EBITDA margin of 32%.
    The board declared a quarterly dividend of $0.24 per share, payable 
    on February 15, 2025 to shareholders of record as of February 1, 2025.
    """
    
    print("\n" + "="*60)
    print("Testing Entity Extractor")
    print("="*60)
    
    entities = extractor.extract(test_text)
    
    print(f"\nFound {len(entities)} entities:\n")
    
    summary = extractor.get_summary(entities)
    print("Entity Summary:")
    for ent_type, count in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {ent_type}: {count}")
    
    print("\n" + extractor.format_entities_for_display(entities))
