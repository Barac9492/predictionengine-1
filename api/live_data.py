#!/usr/bin/env python3
"""
Live Data API Endpoint
Fetches real-time market data and generates fresh predictions
"""

import sys
import os
from datetime import datetime, timedelta
import json
from typing import Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

TARGET_STOCKS = ['AAPL', 'GOOGL', 'NVDA', 'TSLA']

class LiveMarketAnalyzer:
    """Real-time market data analyzer"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes
        self._cache = {}
        
    def get_live_quote(self, symbol: str) -> Dict:
        """Get real-time quote data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="5d", interval="1d")
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price - prev_close) / prev_close) * 100
            
            # Get real-time info
            quote = {
                'symbol': symbol,
                'current_price': round(float(current_price), 2),
                'change_percent': round(float(change_pct), 2),
                'timestamp': datetime.now().isoformat(),
                'market_cap': info.get('marketCap', 0),
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                'high_52w': info.get('fiftyTwoWeekHigh', 0),
                'low_52w': info.get('fiftyTwoWeekLow', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 0)
            }
            
            return quote
            
        except Exception as e:
            print(f"Error getting live quote for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, hist_data: pd.DataFrame) -> Dict:
        """Calculate technical indicators from historical data"""
        if len(hist_data) < 20:
            return {}
            
        try:
            # RSI
            delta = hist_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # Moving averages
            ma_5 = hist_data['Close'].rolling(5).mean()
            ma_20 = hist_data['Close'].rolling(20).mean()
            
            # Momentum
            momentum_5d = hist_data['Close'].pct_change(5)
            
            # Volume
            volume_ma = hist_data['Volume'].rolling(20).mean()
            volume_ratio = hist_data['Volume'] / volume_ma
            
            return {
                'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
                'ma_5': float(ma_5.iloc[-1]) if not pd.isna(ma_5.iloc[-1]) else 0.0,
                'ma_20': float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else 0.0,
                'momentum_5d': float(momentum_5d.iloc[-1]) if not pd.isna(momentum_5d.iloc[-1]) else 0.0,
                'volume_ratio': float(volume_ratio.iloc[-1]) if not pd.isna(volume_ratio.iloc[-1]) else 1.0
            }
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}
    
    def generate_ml_prediction(self, hist_data: pd.DataFrame) -> Dict:
        """Generate ML prediction from recent data"""
        try:
            if len(hist_data) < 50:
                return {'prediction': 0.0, 'confidence': 0.5}
                
            # Simple features
            features = []
            for i in range(5, len(hist_data)):
                row = [
                    hist_data['Close'].iloc[i] / hist_data['Close'].iloc[i-1] - 1,  # 1-day return
                    hist_data['Close'].iloc[i] / hist_data['Close'].iloc[i-5] - 1,  # 5-day return
                    hist_data['Volume'].iloc[i] / hist_data['Volume'].iloc[i-5],    # Volume ratio
                    (hist_data['High'].iloc[i] - hist_data['Low'].iloc[i]) / hist_data['Close'].iloc[i]  # Daily range
                ]
                features.append(row)
            
            # Targets (next day returns)
            targets = []
            for i in range(5, len(hist_data)-1):
                target = hist_data['Close'].iloc[i+1] / hist_data['Close'].iloc[i] - 1
                targets.append(target)
            
            if len(features) != len(targets):
                features = features[:len(targets)]
            
            if len(features) < 20:
                return {'prediction': 0.0, 'confidence': 0.5}
            
            # Train simple model
            X = np.array(features)
            y = np.array(targets)
            
            # Use last 80% for training
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predict on latest data
            latest_features = np.array([features[-1]])
            latest_scaled = scaler.transform(latest_features)
            prediction = model.predict(latest_scaled)[0]
            
            # Estimate confidence (inverse of prediction variance)
            tree_predictions = [tree.predict(latest_scaled)[0] for tree in model.estimators_]
            confidence = 1 / (1 + np.std(tree_predictions) * 10)
            
            return {
                'prediction': float(prediction),
                'confidence': float(min(0.95, max(0.5, confidence)))
            }
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return {'prediction': 0.0, 'confidence': 0.5}
    
    def generate_trading_signal(self, quote: Dict, indicators: Dict, ml_result: Dict) -> Dict:
        """Generate trading signal based on strategy rules"""
        try:
            current_price = quote['current_price']
            rsi = indicators.get('rsi', 50)
            ma_20 = indicators.get('ma_20', current_price)
            momentum_5d = indicators.get('momentum_5d', 0)
            volume_ratio = indicators.get('volume_ratio', 1)
            ml_prediction = ml_result.get('prediction', 0)
            ml_confidence = ml_result.get('confidence', 0.5)
            
            # Apply trading strategy rules
            buy_conditions = [
                current_price > ma_20,  # Uptrend
                45 <= rsi <= 65,        # Momentum, not overbought
                momentum_5d > 0.01,     # Recent strength
                ml_prediction > 0.01,   # ML bullish
                ml_confidence > 0.60,   # Sufficient confidence
                volume_ratio > 1.0      # Volume confirmation
            ]
            
            sell_conditions = [
                rsi > 75,               # Overbought
                current_price < ma_20,  # Trend broken
                momentum_5d < -0.02,    # Weakness
                ml_prediction < -0.015 and ml_confidence > 0.60  # ML bearish
            ]
            
            # Determine action
            if all(buy_conditions):
                action = "BUY"
                confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
                rationale = f"All buy conditions met: Price>${ma_20:.2f}, RSI={rsi:.1f}, momentum={momentum_5d*100:.1f}%"
            elif any(sell_conditions):
                action = "SELL"
                confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
                rationale = f"Sell conditions triggered: RSI={rsi:.1f}, momentum={momentum_5d*100:.1f}%"
            else:
                action = "HOLD"
                confidence = "LOW" if ml_confidence < 0.6 else "MEDIUM"
                rationale = f"No clear signals. RSI={rsi:.1f}, ML pred={ml_prediction*100:.1f}%"
            
            target_price = current_price * (1 + ml_prediction)
            
            return {
                'action': action,
                'confidence': confidence,
                'rationale': rationale,
                'target_price': round(target_price, 2),
                'expected_return_pct': round(ml_prediction * 100, 2),
                'market_metrics': {
                    'rsi': round(rsi, 1),
                    'rsi_signal': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral',
                    'price_vs_ma20': 'Above' if current_price > ma_20 else 'Below',
                    'momentum_5d_pct': round(momentum_5d * 100, 2),
                    'volume_ratio': round(volume_ratio, 2),
                    'ml_prediction_pct': round(ml_prediction * 100, 2),
                    'ml_confidence_pct': round(ml_confidence * 100, 1)
                }
            }
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return {
                'action': 'HOLD',
                'confidence': 'LOW',
                'rationale': f'Error in analysis: {str(e)}',
                'target_price': quote['current_price'],
                'expected_return_pct': 0.0
            }
    
    def analyze_stock_live(self, symbol: str) -> Dict:
        """Complete live analysis for a single stock"""
        try:
            # Get live quote
            quote = self.get_live_quote(symbol)
            if not quote:
                return None
            
            # Get historical data for indicators
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo", interval="1d")
            
            # Calculate indicators
            indicators = self.calculate_technical_indicators(hist)
            
            # Generate ML prediction
            ml_result = self.generate_ml_prediction(hist)
            
            # Generate trading signal
            signal = self.generate_trading_signal(quote, indicators, ml_result)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'quote': quote,
                'current_recommendation': {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z'),
                    'price': quote['current_price'],
                    **signal
                },
                'backtest_summary': {
                    'total_return_pct': 0,
                    'outperformance_pct': 0,
                    'win_rate_pct': 0,
                    'total_trades': 0,
                    'sharpe_ratio': 0
                },
                'strategy': {
                    'name': 'Active Momentum Trading (LIVE)',
                    'description': 'Real-time analysis with live market data'
                },
                'is_live': True
            }
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None

def lambda_handler(event, context):
    """AWS Lambda handler (also works for Vercel)"""
    
    # Enable CORS
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    
    try:
        analyzer = LiveMarketAnalyzer()
        results = {}
        
        for symbol in TARGET_STOCKS:
            result = analyzer.analyze_stock_live(symbol)
            if result:
                results[symbol] = result
        
        # Add summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(results),
            'market_status': 'LIVE' if results else 'ERROR',
            'buy_signals': len([r for r in results.values() if r['current_recommendation']['action'] == 'BUY']),
            'sell_signals': len([r for r in results.values() if r['current_recommendation']['action'] == 'SELL']),
            'hold_signals': len([r for r in results.values() if r['current_recommendation']['action'] == 'HOLD'])
        }
        
        response_data = {
            'success': True,
            'data': results,
            'summary': summary,
            'generated_at': datetime.now().isoformat()
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response_data, default=str)
        }
        
    except Exception as e:
        error_response = {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps(error_response)
        }

# For Vercel
def handler(request):
    """Vercel handler"""
    return lambda_handler({}, {})

if __name__ == "__main__":
    # Test locally
    result = lambda_handler({}, {})
    print(json.dumps(json.loads(result['body']), indent=2))