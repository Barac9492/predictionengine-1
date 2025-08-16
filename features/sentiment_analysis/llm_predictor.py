"""
LLM-based Stock Sentiment Analysis Module
Based on "Can ChatGPT Forecast Stock Price Movements?" research paper
Adapted for use with various LLM APIs (Grok, OpenAI, etc.)
"""

import os
import json
import requests
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

class SentimentScore(Enum):
    """Sentiment scoring based on research paper methodology"""
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0

@dataclass
class NewsHeadline:
    """Structure for news headline data"""
    headline: str
    source: str
    timestamp: datetime
    relevance_score: float
    company: str
    ticker: str

@dataclass
class SentimentAnalysis:
    """Result of sentiment analysis"""
    headline: str
    score: SentimentScore
    confidence: float
    explanation: str
    raw_response: str
    timestamp: datetime

class LLMSentimentAnalyzer:
    """
    Advanced sentiment analyzer using LLM for stock prediction
    Based on academic research methodology
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Initialize sentiment analyzer
        
        Args:
            api_key: API key for LLM service
            model: Model to use (gpt-4, grok-4, claude-3, etc.)
        """
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model
        self.base_url = self._get_base_url(model)
        
    def _get_base_url(self, model: str) -> str:
        """Get API base URL based on model"""
        urls = {
            "grok-4": "https://api.x.ai/v1/chat/completions",
            "gpt-4": "https://api.openai.com/v1/chat/completions",
            "claude-3": "https://api.anthropic.com/v1/messages"
        }
        return urls.get(model, urls["gpt-4"])
    
    def analyze_headline(self, headline: NewsHeadline) -> SentimentAnalysis:
        """
        Analyze a single headline using LLM
        Following research paper prompt structure
        
        Args:
            headline: NewsHeadline object
            
        Returns:
            SentimentAnalysis result
        """
        # Research paper prompt template
        prompt = f"""
        Forget all your previous instructions. Pretend you are a financial expert. 
        You are a financial expert with stock recommendation experience. 
        Answer 'YES' if good news, 'NO' if bad news, or 'UNKNOWN' if uncertain in the first line. 
        Then elaborate with one short and concise sentence on the next line. 
        Is this headline good or bad for the stock price of {headline.company} in the short term? 
        Headline: {headline.headline}
        """
        
        # Make API call
        response = self._call_llm(prompt)
        
        # Parse response
        score, explanation, raw = self._parse_response(response)
        
        # Calculate confidence based on response clarity
        confidence = self._calculate_confidence(response, headline.relevance_score)
        
        return SentimentAnalysis(
            headline=headline.headline,
            score=score,
            confidence=confidence,
            explanation=explanation,
            raw_response=raw,
            timestamp=datetime.now()
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM service"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # Zero temperature for consistency
            "max_tokens": 100
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data:
                return data["choices"][0]["message"]["content"].strip()
            elif "content" in data:  # Claude format
                return data["content"][0]["text"].strip()
            else:
                return "UNKNOWN\nUnable to analyze"
                
        except Exception as e:
            print(f"LLM API error: {e}")
            return "UNKNOWN\nAPI error occurred"
    
    def _parse_response(self, response: str) -> Tuple[SentimentScore, str, str]:
        """Parse LLM response into sentiment score and explanation"""
        lines = response.split('\n')
        
        # Get first line sentiment
        first_line = lines[0].strip().upper() if lines else "UNKNOWN"
        
        # Map to sentiment score
        if "YES" in first_line:
            score = SentimentScore.POSITIVE
        elif "NO" in first_line:
            score = SentimentScore.NEGATIVE
        else:
            score = SentimentScore.NEUTRAL
        
        # Get explanation (second line)
        explanation = lines[1].strip() if len(lines) > 1 else "No explanation provided"
        
        return score, explanation, response
    
    def _calculate_confidence(self, response: str, relevance_score: float) -> float:
        """
        Calculate confidence score based on response quality and relevance
        
        Args:
            response: LLM response text
            relevance_score: News relevance score (0-1)
            
        Returns:
            Confidence score (0-1)
        """
        base_confidence = 0.5
        
        # Clear YES/NO increases confidence
        if "YES" in response.upper() or "NO" in response.upper():
            base_confidence += 0.3
        
        # Longer explanation increases confidence
        if len(response) > 50:
            base_confidence += 0.1
        
        # Factor in relevance score
        base_confidence *= relevance_score
        
        return min(0.95, max(0.1, base_confidence))
    
    def batch_analyze(self, headlines: List[NewsHeadline]) -> List[SentimentAnalysis]:
        """Analyze multiple headlines in batch"""
        results = []
        for headline in headlines:
            analysis = self.analyze_headline(headline)
            results.append(analysis)
        return results

class NewsDataProvider:
    """Provider for news headlines from various sources"""
    
    def __init__(self, source: str = "yfinance"):
        """
        Initialize news provider
        
        Args:
            source: Data source (yfinance, newsapi, scraping, etc.)
        """
        self.source = source
        
    def get_headlines(self, ticker: str, days_back: int = 7) -> List[NewsHeadline]:
        """
        Get recent headlines for a ticker
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            
        Returns:
            List of NewsHeadline objects
        """
        if self.source == "yfinance":
            return self._get_yfinance_news(ticker, days_back)
        else:
            return self._get_mock_news(ticker)
    
    def _get_yfinance_news(self, ticker: str, days_back: int) -> List[NewsHeadline]:
        """Get news from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            headlines = []
            for item in news[:20]:  # Limit to 20 most recent
                headline = NewsHeadline(
                    headline=item.get('title', ''),
                    source=item.get('publisher', 'Unknown'),
                    timestamp=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    relevance_score=0.8,  # Default relevance
                    company=ticker,
                    ticker=ticker
                )
                headlines.append(headline)
            
            return headlines
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._get_mock_news(ticker)
    
    def _get_mock_news(self, ticker: str) -> List[NewsHeadline]:
        """Generate mock news for testing"""
        mock_headlines = {
            "AAPL": [
                "Apple announces record iPhone sales in Q4",
                "Apple faces supply chain disruptions in China",
                "New Apple Vision Pro receives mixed reviews"
            ],
            "GOOGL": [
                "Google unveils breakthrough in quantum computing",
                "Alphabet reports strong cloud revenue growth",
                "Regulatory concerns mount over Google's AI dominance"
            ],
            "NVDA": [
                "NVIDIA's new AI chip exceeds performance expectations",
                "NVIDIA stock soars on data center demand",
                "Competition heats up in GPU market"
            ],
            "TSLA": [
                "Tesla reports record vehicle deliveries",
                "Tesla cuts prices amid growing competition",
                "Elon Musk announces new Tesla factory location"
            ]
        }
        
        headlines = []
        for i, text in enumerate(mock_headlines.get(ticker, ["Generic news about " + ticker])):
            headline = NewsHeadline(
                headline=text,
                source="Mock Source",
                timestamp=datetime.now() - timedelta(days=i),
                relevance_score=0.9 - (i * 0.1),
                company=ticker,
                ticker=ticker
            )
            headlines.append(headline)
        
        return headlines

class SentimentTradingStrategy:
    """
    Trading strategy based on sentiment scores
    Implements long-short portfolio from research paper
    """
    
    def __init__(self, analyzer: LLMSentimentAnalyzer):
        """
        Initialize trading strategy
        
        Args:
            analyzer: LLM sentiment analyzer instance
        """
        self.analyzer = analyzer
        self.news_provider = NewsDataProvider()
        
    def generate_signals(self, tickers: List[str]) -> Dict[str, float]:
        """
        Generate trading signals for multiple tickers
        
        Args:
            tickers: List of stock tickers
            
        Returns:
            Dictionary of ticker -> position size (-1 to 1)
        """
        signals = {}
        
        for ticker in tickers:
            # Get recent headlines
            headlines = self.news_provider.get_headlines(ticker, days_back=7)
            
            if not headlines:
                signals[ticker] = 0
                continue
            
            # Analyze sentiment
            analyses = self.analyzer.batch_analyze(headlines)
            
            # Calculate aggregate score
            weighted_score = self._calculate_weighted_score(analyses)
            
            # Generate signal
            signals[ticker] = self._score_to_signal(weighted_score)
        
        return signals
    
    def _calculate_weighted_score(self, analyses: List[SentimentAnalysis]) -> float:
        """
        Calculate weighted sentiment score
        
        Args:
            analyses: List of sentiment analyses
            
        Returns:
            Weighted score between -1 and 1
        """
        if not analyses:
            return 0
        
        total_weight = 0
        weighted_sum = 0
        
        for analysis in analyses:
            # Weight by confidence and recency
            days_old = (datetime.now() - analysis.timestamp).days
            recency_weight = 1.0 / (1 + days_old * 0.1)  # Decay over time
            
            weight = analysis.confidence * recency_weight
            weighted_sum += analysis.score.value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return weighted_sum / total_weight
    
    def _score_to_signal(self, score: float) -> float:
        """
        Convert sentiment score to trading signal
        
        Args:
            score: Weighted sentiment score
            
        Returns:
            Position size (-1 to 1)
        """
        # Threshold-based approach from paper
        if score > 0.3:
            return 1.0  # Strong buy
        elif score > 0.1:
            return 0.5  # Moderate buy
        elif score < -0.3:
            return -1.0  # Strong sell
        elif score < -0.1:
            return -0.5  # Moderate sell
        else:
            return 0  # Hold/neutral
    
    def backtest(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Backtest strategy performance
        
        Args:
            tickers: List of tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with backtest results
        """
        results = []
        
        for ticker in tickers:
            # Get price data
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            returns = stock_data['Adj Close'].pct_change()
            
            # Generate signals (simplified - would need historical news in practice)
            signal = self.generate_signals([ticker])[ticker]
            
            # Calculate strategy return
            strategy_return = signal * returns.mean()
            
            results.append({
                'ticker': ticker,
                'signal': signal,
                'avg_return': returns.mean(),
                'strategy_return': strategy_return,
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
            })
        
        return pd.DataFrame(results)

# Example usage
def example_sentiment_analysis():
    """Example of using the sentiment analysis system"""
    
    # Initialize analyzer (would need actual API key)
    analyzer = LLMSentimentAnalyzer(model="gpt-4")
    
    # Create mock headline
    headline = NewsHeadline(
        headline="Apple reports record-breaking Q4 earnings, beating analyst expectations",
        source="Reuters",
        timestamp=datetime.now(),
        relevance_score=0.95,
        company="Apple Inc.",
        ticker="AAPL"
    )
    
    # Analyze sentiment (mock result since no actual API key)
    mock_analysis = SentimentAnalysis(
        headline=headline.headline,
        score=SentimentScore.POSITIVE,
        confidence=0.92,
        explanation="Strong earnings beat indicates positive momentum for Apple stock",
        raw_response="YES\nStrong earnings beat indicates positive momentum for Apple stock",
        timestamp=datetime.now()
    )
    
    print(f"Headline: {mock_analysis.headline}")
    print(f"Sentiment: {mock_analysis.score.name}")
    print(f"Confidence: {mock_analysis.confidence:.2%}")
    print(f"Explanation: {mock_analysis.explanation}")
    
    # Trading strategy
    strategy = SentimentTradingStrategy(analyzer)
    signals = strategy.generate_signals(["AAPL", "GOOGL", "NVDA", "TSLA"])
    
    print("\nTrading Signals:")
    for ticker, signal in signals.items():
        action = "BUY" if signal > 0 else "SELL" if signal < 0 else "HOLD"
        print(f"{ticker}: {action} (signal: {signal:.2f})")
    
    return mock_analysis, signals

if __name__ == "__main__":
    analysis, signals = example_sentiment_analysis()
    print("\nSentiment-based trading strategy initialized successfully!")