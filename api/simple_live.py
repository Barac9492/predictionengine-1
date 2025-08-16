"""
Simplified live data endpoint for Vercel
Minimal dependencies - just Python standard library
"""

import json
import random
import math
from datetime import datetime
from http.server import BaseHTTPRequestHandler

TARGET_STOCKS = ['AAPL', 'GOOGL', 'NVDA', 'TSLA']

# Base prices (updated August 16, 2025)
BASE_PRICES = {
    'AAPL': 214.79,
    'GOOGL': 195.81,
    'NVDA': 178.34,
    'TSLA': 318.17
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
                results[symbol] = generate_simple_live_data(symbol)
            
            response = {
                'success': True,
                'data': results,
                'timestamp': datetime.now().isoformat(),
                'total_stocks': len(results),
                'is_live': True,
                'data_source': 'simple_live_simulation'
            }
            
        except Exception as e:
            response = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        self.wfile.write(json.dumps(response, default=str).encode())

def generate_simple_live_data(symbol):
    """Generate realistic mock live data for a stock with minimal dependencies"""
    
    # Get base price and add realistic variation
    base_price = BASE_PRICES[symbol]
    
    # Simulate intraday price movement (±3% max)
    price_variation = random.uniform(-0.03, 0.03)
    current_price = base_price * (1 + price_variation)
    
    # Simulate daily change
    daily_change = random.uniform(-2.5, 2.5)
    
    # Generate basic technical indicators
    rsi = random.uniform(25, 75)  # RSI range
    momentum_5d = random.uniform(-0.06, 0.06)  # ±6% momentum
    volume_ratio = random.uniform(0.6, 1.8)  # Volume vs average
    
    # Simple ML-like prediction (using basic math)
    trend_factor = math.sin(random.uniform(0, 2 * math.pi)) * 0.02
    ml_prediction = trend_factor + momentum_5d * 0.3
    ml_confidence = random.uniform(0.55, 0.92)
    
    # Generate trading signals based on simple rules
    price_above_ma = random.choice([True, False])
    
    # Buy signal logic
    if (rsi < 50 and momentum_5d > 0.015 and ml_prediction > 0.01 and ml_confidence > 0.7):
        action = "BUY"
        confidence = "HIGH" if ml_confidence > 0.85 else "MEDIUM"
        rationale = f"Strong buy signals: RSI {rsi:.1f}, positive momentum, ML confident"
    # Sell signal logic  
    elif (rsi > 70 or momentum_5d < -0.025 or (ml_prediction < -0.015 and ml_confidence > 0.7)):
        action = "SELL"
        confidence = "HIGH" if ml_confidence > 0.85 else "MEDIUM"
        rationale = f"Sell signals detected: RSI {rsi:.1f}, negative momentum or prediction"
    else:
        action = "HOLD"
        confidence = "MEDIUM" if ml_confidence > 0.6 else "LOW"
        rationale = f"Mixed signals, holding position: RSI {rsi:.1f}"
    
    target_price = current_price * (1 + ml_prediction)
    
    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'is_live': True,
        'quote': {
            'current_price': round(current_price, 2),
            'change_percent': round(daily_change, 2),
            'market_cap': random.randint(400, 3500) * 1e9,  # Realistic market caps
            'volume': random.randint(15, 250) * 1e6,  # Realistic volume
            'pe_ratio': round(random.uniform(12, 45), 1)
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
            'total_return_pct': round(random.uniform(-5, 25), 1),
            'outperformance_pct': round(random.uniform(-3, 15), 1),
            'win_rate_pct': round(random.uniform(45, 75), 1),
            'total_trades': random.randint(8, 35),
            'sharpe_ratio': round(random.uniform(0.8, 2.2), 2)
        },
        'strategy': {
            'name': 'Simple Live Trading (Simulated)',
            'description': 'Lightweight real-time market simulation'
        }
    }

# For local testing
if __name__ == "__main__":
    result = generate_simple_live_data('AAPL')
    print(json.dumps(result, indent=2, default=str))