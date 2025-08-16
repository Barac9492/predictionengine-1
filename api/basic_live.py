"""
Basic live data endpoint for Vercel - minimal implementation
Uses Vercel's function format instead of BaseHTTPRequestHandler
"""

import json
import random
import math
from datetime import datetime

TARGET_STOCKS = ['AAPL', 'GOOGL', 'NVDA', 'TSLA']

BASE_PRICES = {
    'AAPL': 231.59,
    'GOOGL': 203.90,
    'NVDA': 180.45,
    'TSLA': 330.56
}

def handler(request):
    """Vercel function handler"""
    
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Content-Type': 'application/json'
    }
    
    try:
        results = {}
        
        for symbol in TARGET_STOCKS:
            results[symbol] = generate_basic_live_data(symbol)
        
        response_data = {
            'success': True,
            'data': results,
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(results),
            'is_live': True,
            'data_source': 'basic_live_simulation'
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

def generate_basic_live_data(symbol):
    """Generate basic live data simulation"""
    
    base_price = BASE_PRICES[symbol]
    
    # Small price movement
    price_variation = random.uniform(-0.025, 0.025)
    current_price = base_price * (1 + price_variation)
    
    # Daily change
    daily_change = random.uniform(-2.0, 2.0)
    
    # Basic indicators
    rsi = random.uniform(30, 70)
    momentum_5d = random.uniform(-0.04, 0.04)
    volume_ratio = random.uniform(0.7, 1.5)
    
    # Simple prediction
    ml_prediction = random.uniform(-0.02, 0.02)
    ml_confidence = random.uniform(0.6, 0.9)
    
    # Trading signal
    if rsi < 45 and momentum_5d > 0.01 and ml_prediction > 0.01:
        action = "BUY"
        confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
    elif rsi > 65 or momentum_5d < -0.02:
        action = "SELL" 
        confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
    else:
        action = "HOLD"
        confidence = "MEDIUM"
    
    rationale = f"Basic signals: RSI {rsi:.1f}, momentum {momentum_5d*100:.1f}%"
    target_price = current_price * (1 + ml_prediction)
    
    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'is_live': True,
        'quote': {
            'current_price': round(current_price, 2),
            'change_percent': round(daily_change, 2),
            'market_cap': random.randint(500, 3000) * 1000000000,
            'volume': random.randint(20, 200) * 1000000,
            'pe_ratio': round(random.uniform(15, 50), 1)
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
                'price_vs_ma20': 'Above' if random.choice([True, False]) else 'Below',
                'momentum_5d_pct': round(momentum_5d * 100, 2),
                'volume_ratio': round(volume_ratio, 2),
                'ml_prediction_pct': round(ml_prediction * 100, 2),
                'ml_confidence_pct': round(ml_confidence * 100, 1)
            }
        },
        'backtest_summary': {
            'total_return_pct': round(random.uniform(0, 20), 1),
            'outperformance_pct': round(random.uniform(0, 12), 1),
            'win_rate_pct': round(random.uniform(50, 75), 1),
            'total_trades': random.randint(10, 30),
            'sharpe_ratio': round(random.uniform(1.0, 2.0), 2)
        },
        'strategy': {
            'name': 'Basic Live Trading (Demo)',
            'description': 'Simplified real-time market simulation'
        }
    }