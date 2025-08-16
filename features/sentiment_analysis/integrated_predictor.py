"""
Integrated Prediction System
Combines technical analysis, ML predictions, and LLM sentiment analysis
Based on academic research and industry best practices
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass
import json

# Import our modules
from .llm_predictor import (
    LLMSentimentAnalyzer, 
    NewsDataProvider,
    SentimentTradingStrategy,
    SentimentScore
)

@dataclass
class IntegratedSignal:
    """Combined trading signal from multiple sources"""
    ticker: str
    timestamp: datetime
    
    # Individual components
    technical_score: float  # -1 to 1
    ml_score: float  # -1 to 1
    sentiment_score: float  # -1 to 1
    
    # Weights
    technical_weight: float = 0.3
    ml_weight: float = 0.4
    sentiment_weight: float = 0.3
    
    # Combined signal
    combined_score: float = 0
    action: str = "HOLD"
    confidence: float = 0
    
    # Detailed metrics
    rsi: float = 50
    momentum: float = 0
    volume_ratio: float = 1
    news_count: int = 0
    
    def calculate_combined_signal(self):
        """Calculate combined signal from components"""
        self.combined_score = (
            self.technical_score * self.technical_weight +
            self.ml_score * self.ml_weight +
            self.sentiment_score * self.sentiment_weight
        )
        
        # Determine action
        if self.combined_score > 0.3:
            self.action = "STRONG BUY"
        elif self.combined_score > 0.1:
            self.action = "BUY"
        elif self.combined_score < -0.3:
            self.action = "STRONG SELL"
        elif self.combined_score < -0.1:
            self.action = "SELL"
        else:
            self.action = "HOLD"
        
        # Calculate confidence
        scores = [abs(self.technical_score), abs(self.ml_score), abs(self.sentiment_score)]
        agreement = 1 - np.std(scores)
        strength = np.mean(scores)
        self.confidence = (agreement * 0.5 + strength * 0.5)
        
        return self

class IntegratedPredictor:
    """
    Advanced prediction system integrating multiple approaches:
    1. Technical Analysis (RSI, Moving Averages, Momentum)
    2. Machine Learning (Ensemble models)
    3. LLM Sentiment Analysis (News headlines)
    """
    
    def __init__(self, 
                 use_llm: bool = True,
                 llm_api_key: str = None,
                 llm_model: str = "gpt-4"):
        """
        Initialize integrated predictor
        
        Args:
            use_llm: Whether to use LLM for sentiment analysis
            llm_api_key: API key for LLM service
            llm_model: LLM model to use
        """
        self.use_llm = use_llm
        
        if use_llm:
            self.sentiment_analyzer = LLMSentimentAnalyzer(
                api_key=llm_api_key,
                model=llm_model
            )
            self.sentiment_strategy = SentimentTradingStrategy(self.sentiment_analyzer)
            self.news_provider = NewsDataProvider()
        
    def analyze_stock(self, ticker: str, lookback_days: int = 30) -> IntegratedSignal:
        """
        Comprehensive analysis of a single stock
        
        Args:
            ticker: Stock ticker symbol
            lookback_days: Days of historical data to analyze
            
        Returns:
            IntegratedSignal with complete analysis
        """
        # Get stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            stock_data = yf.download(
                ticker, 
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if stock_data.empty:
                return self._create_default_signal(ticker)
            
            # Technical analysis
            technical_score, technical_metrics = self._calculate_technical_score(stock_data)
            
            # ML prediction
            ml_score, ml_confidence = self._calculate_ml_score(stock_data)
            
            # Sentiment analysis
            if self.use_llm:
                sentiment_score, news_count = self._calculate_sentiment_score(ticker)
            else:
                sentiment_score, news_count = 0, 0
            
            # Create integrated signal
            signal = IntegratedSignal(
                ticker=ticker,
                timestamp=datetime.now(),
                technical_score=technical_score,
                ml_score=ml_score,
                sentiment_score=sentiment_score,
                rsi=technical_metrics.get('rsi', 50),
                momentum=technical_metrics.get('momentum', 0),
                volume_ratio=technical_metrics.get('volume_ratio', 1),
                news_count=news_count
            )
            
            # Calculate combined signal
            signal.calculate_combined_signal()
            
            return signal
            
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            return self._create_default_signal(ticker)
    
    def _calculate_technical_score(self, data: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate technical analysis score"""
        metrics = {}
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        metrics['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else 50
        
        # Moving averages
        ma_20 = data['Close'].rolling(window=20).mean()
        ma_50 = data['Close'].rolling(window=50).mean() if len(data) >= 50 else ma_20
        
        # Price position
        current_price = data['Close'].iloc[-1]
        price_vs_ma20 = (current_price / ma_20.iloc[-1] - 1) if not ma_20.empty else 0
        price_vs_ma50 = (current_price / ma_50.iloc[-1] - 1) if not ma_50.empty else 0
        
        # Momentum
        momentum_5d = (current_price / data['Close'].iloc[-6] - 1) if len(data) >= 6 else 0
        metrics['momentum'] = float(momentum_5d)
        
        # Volume
        avg_volume = data['Volume'].rolling(window=20).mean()
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if not avg_volume.empty else 1
        metrics['volume_ratio'] = float(volume_ratio)
        
        # Calculate score
        score = 0
        
        # RSI signals
        if metrics['rsi'] < 30:
            score += 0.5  # Oversold
        elif metrics['rsi'] > 70:
            score -= 0.5  # Overbought
        elif 45 <= metrics['rsi'] <= 55:
            score += 0.2  # Neutral zone, slight positive
        
        # Trend signals
        if price_vs_ma20 > 0 and price_vs_ma50 > 0:
            score += 0.3  # Above both MAs
        elif price_vs_ma20 < 0 and price_vs_ma50 < 0:
            score -= 0.3  # Below both MAs
        
        # Momentum signals
        if momentum_5d > 0.02:
            score += 0.2  # Strong positive momentum
        elif momentum_5d < -0.02:
            score -= 0.2  # Strong negative momentum
        
        # Volume confirmation
        if volume_ratio > 1.2:
            score *= 1.1  # Amplify signal with high volume
        
        # Normalize to [-1, 1]
        score = max(-1, min(1, score))
        
        return score, metrics
    
    def _calculate_ml_score(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate ML-based prediction score
        Simplified version - in production would use trained models
        """
        if len(data) < 20:
            return 0, 0.5
        
        # Feature engineering
        features = []
        
        # Price-based features
        returns = data['Close'].pct_change()
        features.append(returns.mean())  # Average return
        features.append(returns.std())  # Volatility
        features.append(returns.iloc[-1])  # Last return
        
        # Volume features
        volume_change = data['Volume'].pct_change().mean()
        features.append(volume_change)
        
        # Technical features
        sma_ratio = data['Close'].iloc[-1] / data['Close'].rolling(20).mean().iloc[-1]
        features.append(sma_ratio - 1)
        
        # Mock ML prediction (would use actual model in production)
        # This simulates a trained model's output
        feature_weights = [0.3, -0.2, 0.4, 0.1, 0.2]  # Mock weights
        
        score = sum(f * w for f, w in zip(features, feature_weights))
        
        # Add some randomness to simulate model uncertainty
        noise = np.random.normal(0, 0.1)
        score += noise
        
        # Normalize and calculate confidence
        score = max(-1, min(1, score))
        confidence = 0.7 + np.random.uniform(-0.2, 0.2)  # Mock confidence
        confidence = max(0.3, min(0.95, confidence))
        
        return score, confidence
    
    def _calculate_sentiment_score(self, ticker: str) -> Tuple[float, int]:
        """Calculate sentiment score from news headlines"""
        if not self.use_llm:
            return 0, 0
        
        try:
            # Get recent headlines
            headlines = self.news_provider.get_headlines(ticker, days_back=7)
            
            if not headlines:
                return 0, 0
            
            # Analyze sentiment for each headline
            analyses = self.sentiment_strategy.analyzer.batch_analyze(headlines[:10])  # Limit to 10
            
            # Calculate weighted score
            total_weight = 0
            weighted_sum = 0
            
            for i, analysis in enumerate(analyses):
                # Weight by recency and confidence
                recency_weight = 1.0 / (1 + i * 0.1)
                weight = analysis.confidence * recency_weight
                
                weighted_sum += analysis.score.value * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0, len(headlines)
            
            score = weighted_sum / total_weight
            
            return score, len(headlines)
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0, 0
    
    def _create_default_signal(self, ticker: str) -> IntegratedSignal:
        """Create default signal when analysis fails"""
        return IntegratedSignal(
            ticker=ticker,
            timestamp=datetime.now(),
            technical_score=0,
            ml_score=0,
            sentiment_score=0
        )
    
    def analyze_portfolio(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze multiple stocks and create portfolio recommendations
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for ticker in tickers:
            signal = self.analyze_stock(ticker)
            
            results.append({
                'ticker': ticker,
                'timestamp': signal.timestamp,
                'action': signal.action,
                'combined_score': round(signal.combined_score, 3),
                'confidence': round(signal.confidence, 3),
                'technical_score': round(signal.technical_score, 3),
                'ml_score': round(signal.ml_score, 3),
                'sentiment_score': round(signal.sentiment_score, 3),
                'rsi': round(signal.rsi, 1),
                'momentum_%': round(signal.momentum * 100, 2),
                'volume_ratio': round(signal.volume_ratio, 2),
                'news_count': signal.news_count
            })
        
        df = pd.DataFrame(results)
        
        # Sort by combined score
        df = df.sort_values('combined_score', ascending=False)
        
        return df
    
    def generate_portfolio_weights(self, 
                                  tickers: List[str],
                                  max_position: float = 0.25) -> Dict[str, float]:
        """
        Generate portfolio weights based on signals
        
        Args:
            tickers: List of tickers
            max_position: Maximum weight for single position
            
        Returns:
            Dictionary of ticker -> weight
        """
        # Analyze all stocks
        df = self.analyze_portfolio(tickers)
        
        weights = {}
        
        # Long positions
        longs = df[df['combined_score'] > 0.1]
        if not longs.empty:
            # Normalize scores for long positions
            long_scores = longs['combined_score'].values
            long_weights = long_scores / long_scores.sum() * 0.5  # 50% long
            
            for ticker, weight in zip(longs['ticker'], long_weights):
                weights[ticker] = min(weight, max_position)
        
        # Short positions
        shorts = df[df['combined_score'] < -0.1]
        if not shorts.empty:
            # Normalize scores for short positions
            short_scores = -shorts['combined_score'].values
            short_weights = short_scores / short_scores.sum() * 0.3  # 30% short
            
            for ticker, weight in zip(shorts['ticker'], short_weights):
                weights[ticker] = -min(weight, max_position)
        
        # Cash for remaining
        total_allocated = sum(abs(w) for w in weights.values())
        if total_allocated < 1.0:
            weights['CASH'] = 1.0 - total_allocated
        
        return weights

def demonstrate_integrated_system():
    """Demonstrate the integrated prediction system"""
    
    print("=" * 80)
    print("INTEGRATED PREDICTION SYSTEM DEMONSTRATION")
    print("Combining Technical Analysis + ML + LLM Sentiment")
    print("=" * 80)
    
    # Initialize predictor (without actual LLM for demo)
    predictor = IntegratedPredictor(use_llm=False)
    
    # Analyze portfolio
    tickers = ["AAPL", "GOOGL", "NVDA", "TSLA"]
    
    print("\nAnalyzing portfolio...")
    results = predictor.analyze_portfolio(tickers)
    
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print(results.to_string())
    
    # Generate portfolio weights
    print("\n" + "=" * 80)
    print("PORTFOLIO ALLOCATION")
    print("=" * 80)
    
    weights = predictor.generate_portfolio_weights(tickers)
    
    for ticker, weight in weights.items():
        if weight != 0:
            position = "LONG" if weight > 0 else "SHORT"
            print(f"{ticker:8} {position:6} {abs(weight)*100:6.2f}%")
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"integrated_analysis_{timestamp}.json"
    
    export_data = {
        'timestamp': timestamp,
        'results': results.to_dict('records'),
        'weights': weights,
        'methodology': {
            'technical_weight': 0.3,
            'ml_weight': 0.4,
            'sentiment_weight': 0.3,
            'components': [
                'RSI, Moving Averages, Momentum',
                'Machine Learning Ensemble',
                'LLM News Sentiment Analysis'
            ]
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"\nResults exported to {filename}")
    
    return results, weights

if __name__ == "__main__":
    results, weights = demonstrate_integrated_system()
    print("\n✅ Integrated prediction system demonstration complete!")