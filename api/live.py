"""
Simple live data endpoint for Vercel
"""

import json
from http.server import BaseHTTPRequestHandler
from datetime import datetime
import sys
import os

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    DEPENDENCIES_AVAILABLE = False

TARGET_STOCKS = ['AAPL', 'GOOGL', 'NVDA', 'TSLA']

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            if not DEPENDENCIES_AVAILABLE:
                raise Exception("Required dependencies not available")
            
            results = {}
            
            for symbol in TARGET_STOCKS:
                try:
                    stock_data = get_live_stock_data(symbol)
                    if stock_data:
                        results[symbol] = stock_data
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
            
            response = {
                'success': True,
                'data': results,
                'timestamp': datetime.now().isoformat(),
                'total_stocks': len(results),
                'is_live': True
            }
            
        except Exception as e:
            response = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'fallback_available': True
            }
        
        self.wfile.write(json.dumps(response, default=str).encode())

def get_live_stock_data(symbol):
    """Get live data for a single stock"""
    try:
        # Get current quote
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period="5d")
        
        if hist.empty:
            return None
        
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close) * 100
        
        # Calculate simple technical indicators
        if len(hist) >= 14:
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            rsi = float(100 - (100 / (1 + rs.iloc[-1])))
        else:
            rsi = 50.0
        
        # Moving averages
        ma_20 = float(hist['Close'].rolling(20).mean().iloc[-1]) if len(hist) >= 20 else current_price
        
        # Momentum
        momentum_5d = float((current_price / hist['Close'].iloc[-5] - 1)) if len(hist) > 5 else 0.0
        
        # Volume
        avg_volume = float(hist['Volume'].mean())
        current_volume = float(hist['Volume'].iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Simple ML prediction (basic momentum model)
        if len(hist) >= 10:
            returns = hist['Close'].pct_change().dropna()
            recent_trend = float(returns.tail(5).mean())
            volatility = float(returns.std())
            
            # Simple prediction based on recent trend and momentum
            ml_prediction = recent_trend * 0.5 + momentum_5d * 0.3
            ml_confidence = max(0.5, 1 - volatility * 5)
        else:
            ml_prediction = 0.0
            ml_confidence = 0.5
        
        # Generate trading signal
        price_above_ma = current_price > ma_20
        rsi_neutral = 45 <= rsi <= 65
        positive_momentum = momentum_5d > 0.01
        ml_bullish = ml_prediction > 0.01 and ml_confidence > 0.6
        
        if price_above_ma and rsi_neutral and positive_momentum and ml_bullish:
            action = "BUY"
            confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
        elif rsi > 75 or (not price_above_ma and momentum_5d < -0.02):
            action = "SELL" 
            confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
        else:
            action = "HOLD"
            confidence = "MEDIUM" if ml_confidence > 0.6 else "LOW"
        
        rationale = f"Price ${current_price:.2f}, RSI {rsi:.1f}, momentum {momentum_5d*100:.1f}%"
        target_price = current_price * (1 + ml_prediction)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'is_live': True,
            'quote': {
                'current_price': round(current_price, 2),
                'change_percent': round(change_pct, 2),
                'market_cap': info.get('marketCap', 0),
                'volume': int(current_volume),
                'pe_ratio': info.get('trailingPE', 0)
            },
            'current_recommendation': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'price': round(current_price, 2),
                'action': action,
                'confidence': confidence,
                'rationale': rationale,
                'target_price': round(target_price, 2),
                'expected_return_pct': round(ml_prediction * 100, 2),
                'market_metrics': {
                    'rsi': round(rsi, 1),
                    'rsi_signal': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral',
                    'price_vs_ma20': 'Above' if price_above_ma else 'Below',
                    'momentum_5d_pct': round(momentum_5d * 100, 2),
                    'volume_ratio': round(volume_ratio, 2),
                    'ml_prediction_pct': round(ml_prediction * 100, 2),
                    'ml_confidence_pct': round(ml_confidence * 100, 1)
                }
            },
            'backtest_summary': {
                'total_return_pct': 0,
                'outperformance_pct': 0,
                'win_rate_pct': 0,
                'total_trades': 0,
                'sharpe_ratio': 0
            },
            'strategy': {
                'name': 'Live Momentum Trading',
                'description': 'Real-time market analysis'
            }
        }
        
    except Exception as e:
        print(f"Error getting data for {symbol}: {e}")
        return None

# For local testing
if __name__ == "__main__":
    try:
        result = get_live_stock_data('AAPL')
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Test failed: {e}")