"""
Risk Factor Detection Module
Identifies and categorizes risk-related language in SEC filings.

Supports FinBERT-enhanced severity scoring for more accurate
risk assessment beyond keyword-based detection.

Author: Shyam Sridhar
CSCI S-89B Final Project
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


class RiskDetector:
    """
    Detects and categorizes risk factors in financial documents.
    
    Scans text for risk-related keywords and phrases, categorizing them
    into regulatory, financial, operational, market, and pandemic risks.
    Optionally uses FinBERT for sentiment-based severity assessment.
    """
    
    # Risk patterns organized by category
    RISK_PATTERNS = {
        'regulatory': [
            r'regulatory\s+(?:risk|change|compliance|requirement)',
            r'government\s+regulation',
            r'legal\s+proceedings?',
            r'litigation',
            r'(?:SEC|FDA|FTC)\s+(?:investigation|inquiry|action)',
            r'antitrust',
            r'compliance\s+(?:risk|issue|failure)',
            r'regulatory\s+(?:approval|clearance)',
            r'legislative\s+change',
            r'tax\s+(?:law|regulation)\s+change'
        ],
        'financial': [
            r'material\s+adverse\s+effect',
            r'liquidity\s+(?:risk|concerns?|issues?)',
            r'credit\s+(?:risk|exposure)',
            r'debt\s+(?:covenant|obligation|level)',
            r'impairment\s+(?:charge|loss)',
            r'write-?(?:off|down)',
            r'default(?:ed)?',
            r'bankruptcy',
            r'insolvency',
            r'going\s+concern',
            r'cash\s+flow\s+(?:constraint|shortfall)',
            r'refinancing\s+risk',
            r'interest\s+rate\s+(?:risk|exposure|fluctuation)'
        ],
        'operational': [
            r'supply\s+chain\s+(?:disruption|risk|issue)',
            r'cybersecurity\s+(?:risk|threat|breach|incident)',
            r'data\s+(?:breach|security)',
            r'key\s+(?:personnel|employee|executive)\s+(?:loss|departure|retention)',
            r'business\s+(?:interruption|disruption)',
            r'system\s+(?:failure|outage)',
            r'intellectual\s+property\s+(?:risk|infringement)',
            r'product\s+(?:recall|liability|defect)',
            r'manufacturing\s+(?:issue|problem|disruption)',
            r'quality\s+(?:issue|problem|control)'
        ],
        'market': [
            r'competition|competitive\s+pressure',
            r'market\s+(?:volatility|uncertainty|risk|decline)',
            r'economic\s+(?:downturn|recession|slowdown|uncertainty)',
            r'foreign\s+(?:currency|exchange)\s+(?:risk|fluctuation|exposure)',
            r'commodity\s+(?:price|cost)\s+(?:volatility|fluctuation)',
            r'customer\s+(?:concentration|dependency)',
            r'demand\s+(?:decline|reduction|uncertainty)',
            r'pricing\s+pressure',
            r'market\s+share\s+(?:loss|decline)',
            r'technological\s+(?:change|disruption|obsolescence)'
        ],
        'pandemic': [
            r'COVID-?19',
            r'pandemic',
            r'public\s+health\s+(?:crisis|emergency)',
            r'outbreak',
            r'quarantine',
            r'social\s+distancing',
            r'workforce\s+(?:disruption|safety)',
            r'remote\s+work(?:ing)?'
        ],
        'geopolitical': [
            r'geopolitical\s+(?:risk|tension|uncertainty)',
            r'trade\s+(?:war|dispute|tension|restriction)',
            r'tariff',
            r'sanction',
            r'political\s+(?:instability|risk|uncertainty)',
            r'war|conflict|military\s+action',
            r'terrorism',
            r'civil\s+unrest'
        ],
        'climate': [
            r'climate\s+(?:change|risk)',
            r'environmental\s+(?:regulation|liability|risk)',
            r'carbon\s+(?:emission|footprint|tax)',
            r'natural\s+disaster',
            r'extreme\s+weather',
            r'sustainability\s+(?:risk|requirement)',
            r'ESG\s+(?:risk|requirement)'
        ]
    }
    
    # Risk severity indicators
    SEVERITY_INDICATORS = {
        'high': [
            r'significant(?:ly)?',
            r'material(?:ly)?',
            r'substantial(?:ly)?',
            r'severe(?:ly)?',
            r'critical(?:ly)?',
            r'major'
        ],
        'medium': [
            r'moderate(?:ly)?',
            r'considerable|considerably',
            r'notable|notably'
        ],
        'low': [
            r'minor',
            r'slight(?:ly)?',
            r'limited',
            r'minimal(?:ly)?'
        ]
    }
    
    # Category descriptions for UI
    CATEGORY_DESCRIPTIONS = {
        'regulatory': 'Legal, compliance, and government regulation risks',
        'financial': 'Liquidity, credit, and financial health risks',
        'operational': 'Business operations, supply chain, and cybersecurity risks',
        'market': 'Competition, economic, and market condition risks',
        'pandemic': 'Public health and pandemic-related risks',
        'geopolitical': 'Political, trade, and international risks',
        'climate': 'Environmental, climate, and ESG risks'
    }
    
    # Category icons for UI
    CATEGORY_ICONS = {
        'regulatory': '⚖️',
        'financial': '💰',
        'operational': '⚙️',
        'market': '📈',
        'pandemic': '🦠',
        'geopolitical': '🌍',
        'climate': '🌡️'
    }
    
    def __init__(self, finbert_pipeline: Optional[Any] = None):
        """
        Initialize the risk detector with compiled patterns.
        
        Args:
            finbert_pipeline: Optional FinBERT pipeline for enhanced severity scoring.
                             When provided, uses FinBERT sentiment to assess risk severity
                             (negative sentiment = higher risk severity).
        """
        self.finbert = finbert_pipeline
        self._use_finbert = finbert_pipeline is not None
        
        # Compile patterns for efficiency
        self._compiled_patterns = {}
        for category, patterns in self.RISK_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        # Compile severity patterns
        self._severity_patterns = {
            level: [re.compile(p, re.IGNORECASE) for p in patterns]
            for level, patterns in self.SEVERITY_INDICATORS.items()
        }
        
        mode = "FinBERT-enhanced" if self._use_finbert else "keyword-based"
        print(f"  ✓ Risk detector ready ({mode})")
    
    def detect(self, text: str) -> Dict[str, List[Dict]]:
        """
        Detect risk factors in text.
        
        Args:
            text: Document text to analyze
            
        Returns:
            Dictionary with risk categories and their matches.
            When FinBERT is enabled, includes sentiment-based severity scores.
        """
        results = defaultdict(list)
        
        # Search for each category
        for category, patterns in self._compiled_patterns.items():
            category_matches = []
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get context around match
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()
                    
                    # Determine severity
                    if self._use_finbert:
                        # Use FinBERT for severity assessment
                        severity, severity_score = self._assess_severity_with_finbert(context)
                        finbert_sentiment = severity.get('finbert_sentiment', 'neutral')
                        finbert_score = severity.get('finbert_score', 0.5)
                        severity_level = severity['level']
                    else:
                        # Use keyword-based severity assessment
                        severity_level = self._assess_severity(context)
                        severity_score = {'high': 0.9, 'medium': 0.6, 'low': 0.3}.get(severity_level, 0.6)
                        finbert_sentiment = None
                        finbert_score = None
                    
                    match_dict = {
                        'text': match.group(),
                        'context': context,
                        'severity': severity_level,
                        'severity_score': severity_score,
                        'position': match.start()
                    }
                    
                    if self._use_finbert:
                        match_dict['finbert_sentiment'] = finbert_sentiment
                        match_dict['finbert_score'] = finbert_score
                    
                    category_matches.append(match_dict)
            
            if category_matches:
                # Deduplicate by text
                seen = set()
                unique_matches = []
                for m in category_matches:
                    if m['text'].lower() not in seen:
                        seen.add(m['text'].lower())
                        unique_matches.append(m)
                
                results[category] = unique_matches
        
        return dict(results)
    
    def _assess_severity_with_finbert(self, context: str) -> Tuple[Dict, float]:
        """
        Assess risk severity using FinBERT sentiment analysis.
        
        Negative sentiment indicates higher risk severity.
        
        Args:
            context: Text surrounding the risk mention
            
        Returns:
            Tuple of (severity dict, severity score 0-1)
        """
        try:
            result = self.finbert(context[:512])[0]
            sentiment = result['label'].lower()
            score = result['score']
            
            # Map sentiment to severity
            # Negative sentiment = high severity (the risk is being emphasized negatively)
            # Neutral = medium (factual mention)
            # Positive = low (risk is being mitigated or minimized)
            if sentiment == 'negative':
                level = 'high'
                severity_score = 0.7 + (0.3 * score)
            elif sentiment == 'neutral':
                level = 'medium'
                severity_score = 0.4 + (0.3 * score)
            else:  # positive
                level = 'low'
                severity_score = 0.1 + (0.2 * score)
            
            return {
                'level': level,
                'finbert_sentiment': sentiment,
                'finbert_score': score
            }, severity_score
            
        except Exception:
            # Fallback to keyword-based
            level = self._assess_severity(context)
            score = {'high': 0.9, 'medium': 0.6, 'low': 0.3}.get(level, 0.6)
            return {'level': level, 'finbert_sentiment': None, 'finbert_score': None}, score
    
    def _assess_severity(self, context: str) -> str:
        """
        Assess the severity of a risk based on surrounding context.
        
        Args:
            context: Text surrounding the risk mention
            
        Returns:
            Severity level: 'high', 'medium', or 'low'
        """
        for level in ['high', 'medium', 'low']:
            for pattern in self._severity_patterns[level]:
                if pattern.search(context):
                    return level
        return 'medium'  # Default to medium if no indicators found
    
    def get_risk_summary(self, risks: Dict[str, List[Dict]]) -> Dict:
        """
        Generate a summary of detected risks.
        
        Args:
            risks: Output from detect() method
            
        Returns:
            Summary dictionary with counts and severity breakdown
        """
        summary = {
            'total_risks': 0,
            'categories': {},
            'severity_breakdown': {'high': 0, 'medium': 0, 'low': 0},
            'top_categories': []
        }
        
        for category, matches in risks.items():
            count = len(matches)
            summary['total_risks'] += count
            
            # Count severities
            severity_counts = {'high': 0, 'medium': 0, 'low': 0}
            for m in matches:
                severity_counts[m['severity']] += 1
                summary['severity_breakdown'][m['severity']] += 1
            
            summary['categories'][category] = {
                'count': count,
                'icon': self.CATEGORY_ICONS.get(category, '⚠️'),
                'description': self.CATEGORY_DESCRIPTIONS.get(category, ''),
                'severity_breakdown': severity_counts,
                'examples': [m['text'] for m in matches[:3]]
            }
        
        # Sort categories by count
        sorted_cats = sorted(
            summary['categories'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        summary['top_categories'] = [cat for cat, _ in sorted_cats[:5]]
        
        return summary
    
    def get_risk_score(self, risks: Dict[str, List[Dict]]) -> Tuple[float, str]:
        """
        Calculate an overall risk score.
        
        Args:
            risks: Output from detect() method
            
        Returns:
            Tuple of (score 0-100, risk level label)
        """
        if not risks:
            return 0.0, 'Low'
        
        if self._use_finbert:
            # Use FinBERT severity scores for more accurate scoring
            total_score = 0
            total_count = 0
            
            for matches in risks.values():
                for m in matches:
                    total_score += m.get('severity_score', 0.6)
                    total_count += 1
            
            if total_count == 0:
                return 0.0, 'Low'
            
            # Average severity score, scaled to 0-100
            avg_score = (total_score / total_count) * 100
            
            # Also factor in quantity of risks
            quantity_factor = min(1.0, total_count / 20)  # Cap at 20 risks
            score = avg_score * 0.7 + (quantity_factor * 100) * 0.3
            score = min(100, score)
        else:
            # Use keyword-based severity weights
            severity_weights = {'high': 3, 'medium': 2, 'low': 1}
            
            total_weighted = 0
            total_count = 0
            
            for matches in risks.values():
                for m in matches:
                    total_weighted += severity_weights[m['severity']]
                    total_count += 1
            
            if total_count == 0:
                return 0.0, 'Low'
            
            # Normalize to 0-100 scale
            max_expected = 50 * 3
            score = min(100, (total_weighted / max_expected) * 100)
        
        # Determine risk level
        if score >= 60:
            level = 'High'
        elif score >= 30:
            level = 'Medium'
        else:
            level = 'Low'
        
        return score, level
    
    def format_for_display(self, risks: Dict[str, List[Dict]]) -> str:
        """
        Format risk detection results for markdown display.
        
        Args:
            risks: Output from detect() method
            
        Returns:
            Markdown-formatted string
        """
        if not risks:
            return "✅ **No significant risk factors detected.**"
        
        summary = self.get_risk_summary(risks)
        score, level = self.get_risk_score(risks)
        
        # Build markdown output
        output = []
        output.append(f"## ⚠️ Risk Analysis\n")
        output.append(f"**Overall Risk Score**: {score:.0f}/100 ({level})\n")
        output.append(f"**Total Risk Mentions**: {summary['total_risks']}\n\n")
        
        # Severity breakdown
        output.append("### Severity Breakdown\n")
        sb = summary['severity_breakdown']
        output.append(f"- 🔴 High: {sb['high']}\n")
        output.append(f"- 🟡 Medium: {sb['medium']}\n")
        output.append(f"- 🟢 Low: {sb['low']}\n\n")
        
        # Categories
        output.append("### Risk Categories\n")
        for category in summary['top_categories']:
            info = summary['categories'][category]
            icon = info['icon']
            count = info['count']
            examples = ', '.join(info['examples'][:3])
            output.append(f"\n**{icon} {category.title()}** ({count} mentions)\n")
            output.append(f"- Examples: {examples}\n")
        
        return ''.join(output)


if __name__ == "__main__":
    # Test the risk detector
    detector = RiskDetector()
    
    sample_text = """
    The Company faces significant regulatory risk in its key markets. 
    Government regulation of the technology sector continues to increase.
    We may experience material adverse effects from supply chain disruptions.
    The COVID-19 pandemic has created substantial uncertainty in our operations.
    Competition in our markets is intense and we face pricing pressure.
    Foreign currency fluctuations could impact our international revenues.
    Cybersecurity threats pose a significant operational risk.
    Climate change regulations may require substantial capital investments.
    """
    
    risks = detector.detect(sample_text)
    print(detector.format_for_display(risks))
