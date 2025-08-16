# api/index.py
"""
Simple API endpoint for Vercel deployment
"""

def handler(request):
    """Basic handler for Vercel serverless function"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': {
            'message': 'Genius Prediction Engine API',
            'version': '2.0.0',
            'status': 'operational',
            'endpoints': {
                'github': 'https://github.com/Barac9492/predictionengine-1',
                'documentation': 'https://github.com/Barac9492/predictionengine-1#readme',
                'api_reference': 'https://github.com/Barac9492/predictionengine-1/blob/main/docs/API_REFERENCE.md'
            },
            'description': 'AI-powered stock prediction engine with noise-resilient trading guidance',
            'features': [
                'Probabilistic predictions with uncertainty quantification',
                'Buy/Hold/Sell trading guidance',
                'Self-learning reinforcement learning',
                'Automated proxy discovery',
                'Comprehensive backtesting'
            ],
            'note': 'This is a serverless API endpoint. For the full application, please clone the repository and run locally.'
        }
    }