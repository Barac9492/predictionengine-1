"""
Mock live data endpoint that simulates real-time data
Used as fallback when yfinance is not available
"""

import json
from http.server import BaseHTTPRequestHandler
from datetime import datetime
import random
import math

TARGET_STOCKS = ['AAPL', 'GOOGL', 'NVDA', 'TSLA']

# Base prices (approximate current market prices)
BASE_PRICES = {
    'AAPL': 231.59,
    'GOOGL': 203.90,
    'NVDA': 180.45,
    'TSLA': 330.56
}

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        try:
            results = {}
            
            for symbol in TARGET_STOCKS:
                results[symbol] = generate_mock_live_data(symbol)
            
            response = {
                'success': True,
                'data': results,
                'timestamp': datetime.now().isoformat(),
                'total_stocks': len(results),
                'is_live': True,
                'data_source': 'mock_live_simulation'
            }
            
        except Exception as e:
            response = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        self.wfile.write(json.dumps(response, default=str).encode())

def generate_mock_live_data(symbol):
    """Generate realistic mock live data for a stock"""
    
    # Get base price and add some realistic variation
    base_price = BASE_PRICES[symbol]
    
    # Simulate market hours price movement (small random changes)
    price_variation = random.uniform(-0.03, 0.03)  # ±3% max
    current_price = base_price * (1 + price_variation)
    
    # Simulate daily change
    daily_change = random.uniform(-2.0, 2.0)  # ±2% daily change
    
    # Generate technical indicators
    rsi = random.uniform(30, 70)  # Realistic RSI range
    momentum_5d = random.uniform(-0.05, 0.05)  # ±5% momentum
    volume_ratio = random.uniform(0.7, 1.5)  # Volume vs average
    
    # Generate ML prediction
    ml_prediction = random.uniform(-0.02, 0.02)  # ±2% prediction
    ml_confidence = random.uniform(0.6, 0.9)  # 60-90% confidence
    
    # Determine action based on indicators
    price_above_ma = random.choice([True, False])
    
    if rsi < 40 and momentum_5d > 0.01 and ml_prediction > 0.01:
        action = "BUY"
        confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
    elif rsi > 65 or momentum_5d < -0.02:
        action = "SELL"
        confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
    else:
        action = "HOLD"
        confidence = "MEDIUM"
    
    rationale = f"Simulated live data: RSI {rsi:.1f}, momentum {momentum_5d*100:.1f}%"
    target_price = current_price * (1 + ml_prediction)
    
    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'is_live': True,
        'quote': {
            'current_price': round(current_price, 2),
            'change_percent': round(daily_change, 2),
            'market_cap': random.randint(500, 4000) * 1e9,  # Realistic market caps
            'volume': random.randint(20, 200) * 1e6,  # Realistic volume
            'pe_ratio': random.uniform(15, 60)
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
            'name': 'Live Mock Trading (Demo)',
            'description': 'Simulated real-time market analysis'
        }
    }

# For local testing
if __name__ == "__main__":
    result = generate_mock_live_data('AAPL')
    print(json.dumps(result, indent=2, default=str))