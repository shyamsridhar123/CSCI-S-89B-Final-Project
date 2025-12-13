"""
Financial Document Intelligence Dashboard - Enhanced UX Version
Interactive Gradio dashboard for analyzing SEC filings and financial documents.

CSCI S-89B Final Project - Shyam Sridhar

Features:
- Document Classification (SEC 10-K sections)
- Sentiment Analysis (FinBERT)
- Named Entity Recognition
- Risk Factor Detection
- Financial Metrics Extraction
- Forward-Looking Statement Detection
- Extractive Summarization
- File Upload (PDF/TXT)
- Downloadable Reports

UX Improvements:
- Executive Summary Card with key takeaways
- Text highlighting for entities and risks
- Entity table view (sortable)
- Analysis toggles for selective processing
- Accordion layout for better navigation
- Actionable error messages

Usage:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
import json
import re
import html as html_module
import tempfile
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.document_classifier import DocumentClassifier
from src.sentiment_analyzer import SentimentAnalyzer
from src.entity_extractor import EntityExtractor
from src.summarizer import ExtractiveSummarizer
from src.risk_detector import RiskDetector
from src.metrics_extractor import MetricsExtractor, ForwardLookingDetector


# CSS for text highlighting
HIGHLIGHT_CSS = """
<style>
.highlight-entity { background-color: #e3f2fd; border-radius: 3px; padding: 1px 3px; }
.highlight-money { background-color: #c8e6c9; border-radius: 3px; padding: 1px 3px; }
.highlight-risk { background-color: #ffcdd2; border-radius: 3px; padding: 1px 3px; }
.highlight-forward { background-color: #fff3e0; border-radius: 3px; padding: 1px 3px; }
.executive-card { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 20px; border-radius: 10px; margin-bottom: 15px;
}
.metric-badge {
    display: inline-block; padding: 5px 10px; margin: 3px;
    border-radius: 15px; font-size: 0.9em;
}
.badge-positive { background-color: #2e7d32; color: #ffffff; font-weight: bold; }
.badge-negative { background-color: #c62828; color: #ffffff; font-weight: bold; }
.badge-neutral { background-color: #1565c0; color: #ffffff; font-weight: bold; }
</style>
"""


class FinancialDashboard:
    """
    Enhanced dashboard class that orchestrates all NLP components.
    
    Provides a unified interface for:
    - Document classification with human-readable labels
    - Sentiment analysis
    - Named entity recognition
    - Risk factor detection
    - Financial metrics extraction
    - Forward-looking statement detection
    - Extractive summarization
    """
    
    def __init__(self, models_path: str = "models"):
        """
        Initialize the Financial Document Intelligence Pipeline.
        
        Args:
            models_path: Path to saved model files
        """
        print("="*60)
        print("Initializing Financial Document Intelligence Pipeline")
        print("="*60)
        
        # Initialize components
        print("\n[1/7] Loading Document Classifier...")
        self.classifier = DocumentClassifier()
        if os.path.exists(os.path.join(models_path, "classifier_model.keras")):
            self.classifier.load(models_path)
            self.classifier_ready = True
        else:
            print("  ⚠ No trained classifier found. Run training notebook first.")
            self.classifier_ready = False
        
        print("\n[2/7] Loading Sentiment Analyzer...")
        self.sentiment = SentimentAnalyzer()
        
        print("\n[3/7] Loading Entity Extractor...")
        self.ner = EntityExtractor()
        
        print("\n[4/7] Loading Risk Detector...")
        self.risk_detector = RiskDetector()
        
        print("\n[5/7] Loading Metrics Extractor...")
        self.metrics_extractor = MetricsExtractor()
        
        print("\n[6/7] Loading Forward-Looking Detector...")
        self.fwd_detector = ForwardLookingDetector()
        
        print("\n[7/7] Loading Summarizer...")
        self.summarizer = ExtractiveSummarizer()
        
        print("\n" + "="*60)
        print("Pipeline Ready!")
        print("="*60 + "\n")
    
    def analyze(self, text: str, progress_callback=None) -> Dict:
        """
        Run full analysis pipeline on a document.
        
        Args:
            text: Document text to analyze
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {}
        
        # Classification
        if progress_callback:
            progress_callback(0.1, "Classifying document...")
        
        if self.classifier_ready:
            section_code, readable_label, confidence = self.classifier.predict_with_label(text)
            results['classification'] = {
                'section_code': section_code,
                'type': readable_label,
                'confidence': confidence,
                'probabilities': self.classifier.predict_proba(text)
            }
        else:
            results['classification'] = {
                'section_code': 'Unknown',
                'type': 'Unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
        
        # Sentiment Analysis
        if progress_callback:
            progress_callback(0.25, "Analyzing sentiment...")
        results['sentiment'] = self.sentiment.analyze(text)
        
        # Entity Extraction
        if progress_callback:
            progress_callback(0.4, "Extracting entities...")
        entities = self.ner.extract(text)
        results['entities'] = entities
        results['entity_summary'] = self.ner.get_summary(entities)
        results['top_entities'] = self.ner.get_top_entities(entities, n=15)
        
        # Risk Detection
        if progress_callback:
            progress_callback(0.55, "Detecting risk factors...")
        risks = self.risk_detector.detect(text)
        results['risks'] = risks
        results['risk_summary'] = self.risk_detector.get_risk_summary(risks)
        results['risk_score'], results['risk_level'] = self.risk_detector.get_risk_score(risks)
        
        # Metrics Extraction
        if progress_callback:
            progress_callback(0.7, "Extracting financial metrics...")
        metrics = self.metrics_extractor.extract(text)
        results['metrics'] = metrics
        results['metrics_summary'] = self.metrics_extractor.get_metrics_summary(metrics)
        
        # Forward-Looking Statements
        if progress_callback:
            progress_callback(0.8, "Detecting forward-looking statements...")
        fwd_statements = self.fwd_detector.detect(text)
        results['forward_looking'] = fwd_statements
        results['forward_looking_summary'] = self.fwd_detector.get_summary(fwd_statements)
        
        # Summarization
        if progress_callback:
            progress_callback(0.9, "Generating summary...")
        results['summary'] = self.summarizer.summarize(text, num_sentences=5)
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return results
    
    def create_sentiment_gauge(self, sentiment_result: Dict) -> go.Figure:
        """Create a sentiment gauge visualization."""
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        if label == 'positive':
            value = 50 + (score * 50)
            color = "green"
        elif label == 'negative':
            value = 50 - (score * 50)
            color = "red"
        else:
            value = 50
            color = "gray"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            title={'text': f"Sentiment: {label.upper()}", 'font': {'size': 20}},
            number={'suffix': '%', 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': '#ffcccc'},
                    {'range': [33, 66], 'color': '#ffffcc'},
                    {'range': [66, 100], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={'color': "black", 'family': "Arial"}
        )
        
        return fig
    
    def create_entity_chart(self, entity_summary: Dict) -> Optional[go.Figure]:
        """Create an entity distribution bar chart."""
        if not entity_summary:
            return None
        
        sorted_items = sorted(entity_summary.items(), key=lambda x: -x[1])
        types = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        color_map = {
            'ORG': '#1f77b4', 'PERSON': '#ff7f0e', 'MONEY': '#2ca02c',
            'PERCENTAGE': '#d62728', 'DATE': '#9467bd', 'GPE': '#8c564b',
            'FISCAL_DATE': '#e377c2', 'TICKER': '#7f7f7f',
            'FINANCIAL_METRIC': '#bcbd22', 'CARDINAL': '#17becf'
        }
        colors = [color_map.get(t, '#333333') for t in types]
        
        fig = go.Figure(data=[
            go.Bar(x=types, y=counts, marker_color=colors, text=counts, textposition='auto')
        ])
        
        fig.update_layout(
            title='Extracted Entities by Type',
            xaxis_title='Entity Type', yaxis_title='Count',
            height=300, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white", plot_bgcolor="white", showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_classification_chart(self, probabilities: Dict) -> Optional[go.Figure]:
        """Create a classification probability bar chart."""
        if not probabilities:
            return None
        
        # Get top 10 classes for readability
        sorted_probs = sorted(probabilities.items(), key=lambda x: -x[1])[:10]
        classes = [self.classifier.get_readable_label(c[0]) for c in sorted_probs]
        probs = [c[1] * 100 for c in sorted_probs]
        
        colors = ['#2ca02c' if p == max(probs) else '#1f77b4' for p in probs]
        
        fig = go.Figure(data=[
            go.Bar(x=classes, y=probs, marker_color=colors,
                   text=[f'{p:.1f}%' for p in probs], textposition='auto')
        ])
        
        fig.update_layout(
            title='Classification Probabilities (Top 10)',
            xaxis_title='Section Type', yaxis_title='Probability (%)',
            yaxis_range=[0, 100], height=300,
            margin=dict(l=20, r=20, t=50, b=100),
            paper_bgcolor="white", plot_bgcolor="white"
        )
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_risk_chart(self, risk_summary: Dict) -> Optional[go.Figure]:
        """Create a risk category chart."""
        if not risk_summary.get('categories'):
            return None
        
        categories = []
        counts = []
        colors = []
        
        color_map = {
            'regulatory': '#1f77b4', 'financial': '#d62728',
            'operational': '#ff7f0e', 'market': '#2ca02c',
            'pandemic': '#9467bd', 'geopolitical': '#8c564b',
            'climate': '#17becf'
        }
        
        for cat, info in risk_summary['categories'].items():
            categories.append(f"{info['icon']} {cat.title()}")
            counts.append(info['count'])
            colors.append(color_map.get(cat, '#333333'))
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=counts, marker_color=colors,
                   text=counts, textposition='auto')
        ])
        
        fig.update_layout(
            title='Risk Factors by Category',
            xaxis_title='Risk Category', yaxis_title='Count',
            height=300, margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white", plot_bgcolor="white"
        )
        
        return fig
    
    def create_risk_gauge(self, score: float, level: str) -> go.Figure:
        """Create a risk score gauge."""
        color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': f"Risk Level: {level}", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color_map.get(level, 'gray')},
                'steps': [
                    {'range': [0, 30], 'color': '#ccffcc'},
                    {'range': [30, 60], 'color': '#ffffcc'},
                    {'range': [60, 100], 'color': '#ffcccc'}
                ]
            }
        ))
        
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        return fig
    
    def generate_report(self, results: Dict, text: str) -> Dict:
        """Generate a downloadable report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'document_length': len(text),
            'classification': results['classification'],
            'sentiment': results['sentiment'],
            'risk_analysis': {
                'score': results['risk_score'],
                'level': results['risk_level'],
                'summary': results['risk_summary']
            },
            'entities': {
                'total': len(results['entities']),
                'by_type': results['entity_summary'],
                'top_entities': results['top_entities']
            },
            'metrics': {
                'total': len(results['metrics']),
                'summary': results['metrics_summary']
            },
            'forward_looking': results['forward_looking_summary'],
            'summary': results['summary']
        }
        return report
    
    def create_executive_summary(self, results: Dict) -> str:
        """Generate executive summary card HTML."""
        sentiment = results['sentiment']['label'].upper()
        sentiment_badge = {
            'POSITIVE': 'badge-positive',
            'NEGATIVE': 'badge-negative',
            'NEUTRAL': 'badge-neutral'
        }.get(sentiment, 'badge-neutral')
        
        # Get top 3 entities
        top_entities = results.get('top_entities', [])[:3]
        entity_str = ', '.join([e['text'] for e in top_entities]) if top_entities else 'None detected'
        
        # Get key metric if available
        key_metrics = results.get('metrics_summary', {}).get('key_metrics', [])
        metric_str = ''
        if key_metrics:
            m = key_metrics[0]
            metric_str = f"<br><strong>Key Metric:</strong> {m['label']}: ${m['value']}"
        
        html_content = f"""
{HIGHLIGHT_CSS}
<div class="executive-card">
    <h3 style="margin-top:0;">📋 Quick Summary</h3>
    <table style="width:100%; color:white;">
        <tr>
            <td><strong>Document Type:</strong></td>
            <td>{results['classification']['type']}</td>
        </tr>
        <tr>
            <td><strong>Confidence:</strong></td>
            <td>{results['classification']['confidence']:.1%}</td>
        </tr>
        <tr>
            <td><strong>Sentiment:</strong></td>
            <td><span class="metric-badge {sentiment_badge}">{sentiment}</span></td>
        </tr>
        <tr>
            <td><strong>Risk Level:</strong></td>
            <td>{results['risk_level']} ({results['risk_score']:.0f}/100)</td>
        </tr>
        <tr>
            <td><strong>Key Entities:</strong></td>
            <td>{entity_str}</td>
        </tr>
        <tr>
            <td><strong>Forward-Looking:</strong></td>
            <td>{results['forward_looking_summary']['total']} statements</td>
        </tr>
    </table>
    {metric_str}
</div>
"""
        return html_content
    
    def create_highlighted_text(self, text: str, results: Dict, max_length: int = 2000) -> str:
        """Create HTML with highlighted entities and risks."""
        # Truncate for display
        display_text = text[:max_length] + ('...' if len(text) > max_length else '')
        display_text = html_module.escape(display_text)
        
        # Collect all highlights with positions
        highlights = []
        
        # Add entity highlights
        for entity in results.get('entities', [])[:50]:
            ent_text = entity.get('text', '')
            ent_type = entity.get('type', '')
            if ent_type == 'MONEY':
                css_class = 'highlight-money'
            else:
                css_class = 'highlight-entity'
            highlights.append((ent_text, css_class, ent_type))
        
        # Add risk highlights
        for category, risk_list in results.get('risks', {}).items():
            for risk in risk_list[:10]:
                risk_text = risk.get('text', risk.get('match', ''))
                if risk_text:
                    highlights.append((risk_text, 'highlight-risk', 'RISK'))
        
        # Sort by length (longest first) to avoid partial replacements
        highlights.sort(key=lambda x: -len(x[0]))
        
        # Apply highlights (simple approach - first match only)
        for term, css_class, label in highlights:
            escaped_term = html_module.escape(term)
            if escaped_term in display_text:
                replacement = f'<span class="{css_class}" title="{label}">{escaped_term}</span>'
                display_text = display_text.replace(escaped_term, replacement, 1)
        
        return f"{HIGHLIGHT_CSS}<div style='font-family: monospace; white-space: pre-wrap; line-height: 1.6;'>{display_text}</div>"
    
    def create_entity_table_html(self, top_entities: List[Dict]) -> str:
        """Create an HTML table for entity display."""
        if not top_entities:
            return "<p><em>No entities detected.</em></p>"
        
        html = '''
        <table style="width:100%; border-collapse: collapse; font-size: 14px;">
            <thead>
                <tr style="background: #f0f0f0;">
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Entity</th>
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Type</th>
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: center;">Count</th>
                </tr>
            </thead>
            <tbody>
        '''
        for ent in top_entities[:20]:
            html += f'''
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">{html_module.escape(ent['text'])}</td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{ent['type']}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{ent.get('count', 1)}</td>
                </tr>
            '''
        html += '</tbody></table>'
        return html


# Global dashboard instance
dashboard = None


def initialize_dashboard():
    """Initialize the dashboard (lazy loading)."""
    global dashboard
    if dashboard is None:
        dashboard = FinancialDashboard()
    return dashboard


def extract_text_from_file(file) -> str:
    """Extract text from uploaded file."""
    if file is None:
        return ""
    
    # Handle both filepath string and file object
    file_path = file.name if hasattr(file, 'name') else file
    print(f"[DEBUG] Extracting text from: {file_path}")
    
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"[DEBUG] File size: {file_size_mb:.1f} MB")
    
    if file_path.lower().endswith('.pdf'):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            print(f"[DEBUG] PDF has {len(reader.pages)} pages")
            
            texts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ''
                texts.append(page_text)
                if i == 0:
                    print(f"[DEBUG] First page chars: {len(page_text)}")
            
            text = ' '.join(texts)
            print(f"[DEBUG] Total extracted chars: {len(text)}")
            
            if len(text.strip()) < 50:
                return "Error: PDF appears to be scanned/image-based. Text extraction yielded minimal content. Try a text-based PDF or paste the text directly."
            
            return text
        except ImportError:
            return "Error: PyPDF2 not installed. Please install with: pip install PyPDF2"
        except Exception as e:
            print(f"[DEBUG] PDF extraction error: {e}")
            return f"Error reading PDF: {str(e)}"
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"[DEBUG] Text file chars: {len(text)}")
            return text
        except Exception as e:
            return f"Error reading file: {str(e)}"


def process_document(text: str,
                      run_classification: bool, run_sentiment: bool,
                      run_entities: bool, run_risk: bool,
                      run_metrics: bool, run_summary: bool,
                      progress=gr.Progress()) -> Tuple:
    """Process a document through the pipeline with selective analysis."""
    
    # Validate input with actionable error
    if not text or len(text.strip()) < 100:
        error_html = """
<div style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px;">
    <h4 style="margin-top:0;">⚠️ Input Required</h4>
    <p>Please provide at least 100 characters of text.</p>
    <p><strong>Options:</strong></p>
    <ul>
        <li>Paste text from an SEC filing or earnings report</li>
        <li>Click a sample document below to try the demo</li>
    </ul>
</div>
"""
        return (error_html, "", None, None, None, None, None, "", "", "", None)
    
    # Initialize dashboard
    dash = initialize_dashboard()
    
    # Check classifier availability with actionable message
    if run_classification and not dash.classifier_ready:
        gr.Warning("Classifier not trained. Run notebooks/02_train_classifier.ipynb first. Skipping classification.")
        run_classification = False
    
    try:
        results = {}
        step = 0
        total_steps = sum([run_classification, run_sentiment, run_entities, run_risk, run_metrics, run_summary])
        if total_steps == 0:
            total_steps = 1
        
        def update_progress(msg):
            nonlocal step
            step += 1
            progress(step / total_steps, desc=msg)
        
        # Classification
        if run_classification:
            update_progress("Classifying document...")
            section_code, readable_label, confidence = dash.classifier.predict_with_label(text)
            results['classification'] = {
                'section_code': section_code,
                'type': readable_label,
                'confidence': confidence,
                'probabilities': dash.classifier.predict_proba(text)
            }
        else:
            results['classification'] = {'section_code': 'N/A', 'type': 'Not analyzed', 'confidence': 0.0, 'probabilities': {}}
        
        # Sentiment Analysis
        if run_sentiment:
            update_progress("Analyzing sentiment...")
            results['sentiment'] = dash.sentiment.analyze(text)
        else:
            results['sentiment'] = {'label': 'neutral', 'score': 0.5}
        
        # Entity Extraction
        if run_entities:
            update_progress("Extracting entities...")
            entities = dash.ner.extract(text)
            results['entities'] = entities
            results['entity_summary'] = dash.ner.get_summary(entities)
            results['top_entities'] = dash.ner.get_top_entities(entities, n=20)
        else:
            results['entities'] = []
            results['entity_summary'] = {}
            results['top_entities'] = []
        
        # Risk Detection
        if run_risk:
            update_progress("Detecting risks...")
            risks = dash.risk_detector.detect(text)
            results['risks'] = risks
            results['risk_summary'] = dash.risk_detector.get_risk_summary(risks)
            results['risk_score'], results['risk_level'] = dash.risk_detector.get_risk_score(risks)
        else:
            results['risks'] = {}
            results['risk_summary'] = {'total_risks': 0, 'severity_breakdown': {'high': 0, 'medium': 0, 'low': 0}, 'categories': {}}
            results['risk_score'], results['risk_level'] = 0.0, 'N/A'
        
        # Metrics Extraction
        if run_metrics:
            update_progress("Extracting metrics...")
            metrics = dash.metrics_extractor.extract(text)
            results['metrics'] = metrics
            results['metrics_summary'] = dash.metrics_extractor.get_metrics_summary(metrics)
            fwd_statements = dash.fwd_detector.detect(text)
            results['forward_looking'] = fwd_statements
            results['forward_looking_summary'] = dash.fwd_detector.get_summary(fwd_statements)
        else:
            results['metrics'] = []
            results['metrics_summary'] = {'key_metrics': []}
            results['forward_looking'] = []
            results['forward_looking_summary'] = {'total': 0, 'high_confidence': 0, 'medium_confidence': 0, 'low_confidence': 0}
        
        # Summarization
        if run_summary:
            update_progress("Generating summary...")
            results['summary'] = dash.summarizer.summarize(text, num_sentences=5)
        else:
            results['summary'] = 'Summary analysis was not selected.'
        
        progress(1.0, desc="Complete!")
        
        # Create executive summary
        exec_summary_html = dash.create_executive_summary(results)
        
        # Create highlighted text
        highlighted_html = dash.create_highlighted_text(text, results)
        
        # Create visualizations
        sentiment_fig = dash.create_sentiment_gauge(results['sentiment']) if run_sentiment else None
        entity_fig = dash.create_entity_chart(results['entity_summary']) if run_entities else None
        class_fig = dash.create_classification_chart(results['classification']['probabilities']) if run_classification else None
        risk_fig = dash.create_risk_chart(results['risk_summary']) if run_risk else None
        risk_gauge = dash.create_risk_gauge(results['risk_score'], results['risk_level']) if run_risk else None
        
        # Create entity table HTML (or empty if not running entities)
        entity_html = dash.create_entity_table_html(results['top_entities']) if run_entities else ""
        
        # Format risk analysis markdown
        risk_md = f"""
### ⚠️ Risk Analysis

**Risk Score**: {results['risk_score']:.0f}/100  
**Risk Level**: {results['risk_level']}  
**Total Risk Mentions**: {results['risk_summary']['total_risks']}

#### Severity Breakdown
- 🔴 High: {results['risk_summary']['severity_breakdown']['high']}
- 🟡 Medium: {results['risk_summary']['severity_breakdown']['medium']}
- 🟢 Low: {results['risk_summary']['severity_breakdown']['low']}
"""
        
        # Format metrics
        metrics_md = "\n### 📊 Key Financial Metrics\n\n"
        for km in results['metrics_summary'].get('key_metrics', [])[:8]:
            unit = km.get('unit') or ''
            metrics_md += f"- {km['label']}: **${km['value']} {unit}**\n"
        if not results['metrics_summary'].get('key_metrics'):
            metrics_md += "*No specific financial metrics detected.*\n"
        
        # Format forward-looking
        fwd_md = f"""
### 🔮 Forward-Looking Statements

**Total Found**: {results['forward_looking_summary']['total']}
- 🟢 High Confidence: {results['forward_looking_summary']['high_confidence']}
- 🟡 Medium Confidence: {results['forward_looking_summary']['medium_confidence']}
- 🔴 Low Confidence: {results['forward_looking_summary']['low_confidence']}
"""
        
        # Generate report
        report = dash.generate_report(results, text)
        report_json = json.dumps(report, indent=2, default=str)
        report_file = tempfile.NamedTemporaryFile(mode='w', suffix='_analysis_report.json', delete=False)
        report_file.write(report_json)
        report_file.close()
        
        return (
            exec_summary_html,
            highlighted_html,
            sentiment_fig,
            entity_fig,
            class_fig,
            risk_gauge,
            risk_fig,
            results['summary'],
            entity_html,
            risk_md + "\n" + metrics_md + "\n" + fwd_md,
            report_file.name
        )
        
    except Exception as e:
        error_html = f"""
<div style="background-color: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 5px;">
    <h4 style="margin-top:0;">❌ Processing Error</h4>
    <p>{html_module.escape(str(e))}</p>
    <p><strong>Suggestions:</strong></p>
    <ul>
        <li>Try with a shorter document</li>
        <li>Check that the text is valid UTF-8</li>
        <li>Ensure all dependencies are installed</li>
    </ul>
</div>
"""
        import traceback
        traceback.print_exc()
        return (error_html, "", None, None, None, None, None, "", None, "", None)


def create_app() -> gr.Blocks:
    """Create the enhanced Gradio application with improved UX."""
    
    with gr.Blocks(
        title="Financial Document Intelligence Pipeline",
        theme=gr.themes.Soft(),
        css="""
        .executive-summary { margin-bottom: 20px; }
        .analysis-toggle { padding: 10px; background: #f5f5f5; border-radius: 8px; }
        """
    ) as app:
        
        # Header
        gr.Markdown("""
# 📊 Financial Document Intelligence Pipeline

**CSCI S-89B Final Project** | Analyze SEC filings and financial documents using NLP
        """)
        
        # Input Section
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="📋 Paste Financial Document Text",
                    placeholder="Paste an SEC filing, earnings report, or financial news article here...\n\nMinimum 100 characters required.",
                    lines=12, max_lines=25
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Analysis Options")
                with gr.Group():
                    run_classification = gr.Checkbox(label="📄 Document Classification", value=True)
                    run_sentiment = gr.Checkbox(label="😊 Sentiment Analysis", value=True)
                    run_entities = gr.Checkbox(label="🏷️ Named Entity Recognition", value=True)
                    run_risk = gr.Checkbox(label="⚠️ Risk Detection", value=True)
                    run_metrics = gr.Checkbox(label="📊 Metrics & Forward-Looking", value=True)
                    run_summary = gr.Checkbox(label="📝 Summarization", value=True)
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", size="lg")
        
        # Executive Summary (always visible at top)
        with gr.Row(elem_classes="executive-summary"):
            exec_summary = gr.HTML(
                value="<div style='padding: 20px; text-align: center; color: #666;'>Paste text and click Analyze to see results</div>",
                label="Executive Summary"
            )
        
        # Results in Accordion
        with gr.Accordion("📄 Document Text (Highlighted)", open=False):
            highlighted_text = gr.HTML(label="Annotated Document")
        
        with gr.Accordion("📊 Classification & Sentiment", open=True):
            with gr.Row():
                with gr.Column():
                    sentiment_plot = gr.Plot(label="Sentiment Analysis")
                with gr.Column():
                    class_plot = gr.Plot(label="Classification Probabilities")
        
        with gr.Accordion("🏷️ Entities", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    entity_plot = gr.Plot(label="Entity Distribution")
                with gr.Column(scale=1):
                    entity_table = gr.HTML(label="Top Entities")
        
        with gr.Accordion("⚠️ Risk Analysis", open=True):
            with gr.Row():
                risk_gauge_plot = gr.Plot(label="Risk Score")
                risk_plot = gr.Plot(label="Risk Categories")
            risk_output = gr.Markdown(label="Risk & Metrics Details")
        
        with gr.Accordion("📝 Summary", open=True):
            summary_output = gr.Textbox(
                label="Executive Summary",
                lines=6,
                interactive=False
            )
        
        with gr.Accordion("📥 Export Report", open=False):
            gr.Markdown("Download the complete analysis as a JSON file for further processing.")
            report_download = gr.File(label="Download JSON Report")
        
        # Sample documents dropdown
        gr.Markdown("### 📚 Load Sample 10-K Filing")
        
        # Load available sample files
        sample_docs = {}
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'sample_docs')
        sample_files = {
            'apple_10k_2024.txt': '🍎 Apple Inc. (FY2024)',
            'microsoft_10k_2024.txt': '🪟 Microsoft Corporation (FY2024)',
            'amazon_10k_2023.txt': '📦 Amazon.com, Inc. (FY2023)',
            'alphabet_10k_2023.txt': '🔤 Alphabet Inc. (FY2023)',
            'meta_10k.txt': '👤 Meta Platforms (10-K)'
        }
        
        for filename, display_name in sample_files.items():
            filepath = os.path.join(sample_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    sample_docs[display_name] = f.read()
        
        sample_choices = list(sample_docs.keys()) if sample_docs else ["No samples available"]
        
        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=sample_choices,
                label="Select a company's 10-K filing",
                value=None,
                interactive=True
            )
            load_sample_btn = gr.Button("📥 Load Sample", size="sm")
        
        def load_sample_document(selection):
            if selection and selection in sample_docs:
                text = sample_docs[selection]
                # Truncate if too long for display (keep first 30K chars)
                if len(text) > 30000:
                    text = text[:30000] + "\n\n[... Document truncated for display. Full analysis will use complete text ...]"
                return text
            return ""
        
        load_sample_btn.click(
            fn=load_sample_document,
            inputs=[sample_dropdown],
            outputs=[input_text]
        )
        
        # Event handlers
        analyze_btn.click(
            fn=process_document,
            inputs=[
                input_text,
                run_classification, run_sentiment, run_entities,
                run_risk, run_metrics, run_summary
            ],
            outputs=[
                exec_summary,
                highlighted_text,
                sentiment_plot,
                entity_plot,
                class_plot,
                risk_gauge_plot,
                risk_plot,
                summary_output,
                entity_table,
                risk_output,
                report_download
            ]
        )
        
        def clear_all():
            return (
                "",  # input_text
                "<div style='padding: 20px; text-align: center; color: #666;'>Paste text and click Analyze to see results</div>",  # exec_summary
                "",  # highlighted_text
                None,  # sentiment_plot
                None,  # entity_plot
                None,  # class_plot
                None,  # risk_gauge_plot
                None,  # risk_plot
                "",  # summary_output
                None,  # entity_table
                "",  # risk_output
                None  # report_download
            )
        
        clear_btn.click(
            fn=clear_all,
            inputs=[],
            outputs=[
                input_text, exec_summary, highlighted_text,
                sentiment_plot, entity_plot, class_plot,
                risk_gauge_plot, risk_plot, summary_output,
                entity_table, risk_output, report_download
            ]
        )
        
        # Footer
        gr.Markdown("""
---
**About this Project**: This Financial Document Intelligence Pipeline was developed 
for CSCI S-89B Introduction to Natural Language Processing. It demonstrates practical 
applications of NLP including document classification using a custom Keras neural network, 
sentiment analysis with FinBERT, named entity recognition with spaCy, risk factor detection,
financial metrics extraction, and extractive summarization.

*Processing time varies based on selected analyses (~1-5 seconds)*
        """)
    
    return app


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Starting Financial Document Intelligence Dashboard")
    print("="*60 + "\n")
    
    # Pre-initialize dashboard
    initialize_dashboard()
    
    # Create and launch app
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
