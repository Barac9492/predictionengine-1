# shared/config/targets.py
# List of target stocks (volatile ones for noise testing)
TARGET_STOCKS = [
    'TSLA',  # Tesla
    'AAPL',  # Apple
    'NVDA',  # Nvidia
    'AMD',   # AMD
    'MSFT',  # Microsoft
    'GOOGL', # Google
    'AMZN',  # Amazon
    'META',  # Meta
    'NFLX',  # Netflix
    'COIN'   # Coinbase (crypto-related volatility)
]

# Example proxies per stock with noise-resilient ideas
PROXY_IDEAS = {
    'TSLA': [
        {'name': 'google_trends_ev', 'source': 'pytrends', 'keywords': ['electric vehicle'], 'smoothing': 'EWMA'},
        {'name': 'lithium_price_index', 'source': 'commodity_data', 'keywords': ['lithium price'], 'smoothing': 'EWMA'},
        {'name': 'battery_tech_news', 'source': 'news_sentiment', 'keywords': ['battery technology'], 'smoothing': 'rolling'},
        {'name': 'ev_sales_global', 'source': 'industry_reports', 'keywords': ['electric vehicle sales'], 'smoothing': 'EWMA'},
        {'name': 'autopilot_mentions', 'source': 'reddit_api', 'keywords': ['Tesla autopilot'], 'smoothing': 'EWMA'}
    ],
    'AAPL': [
        {'name': 'iphone_search_trends', 'source': 'pytrends', 'keywords': ['iPhone'], 'smoothing': 'EWMA'},
        {'name': 'app_store_revenue', 'source': 'app_analytics', 'keywords': ['app store'], 'smoothing': 'EWMA'},
        {'name': 'smartphone_market_share', 'source': 'market_research', 'keywords': ['smartphone market'], 'smoothing': 'rolling'},
        {'name': 'consumer_confidence', 'source': 'fred', 'keywords': ['consumer confidence'], 'smoothing': 'EWMA'},
        {'name': 'tech_spending_index', 'source': 'fred', 'keywords': ['technology spending'], 'smoothing': 'EWMA'}
    ],
    'NVDA': [
        {'name': 'ai_trends', 'source': 'pytrends', 'keywords': ['artificial intelligence'], 'smoothing': 'EWMA'},
        {'name': 'gpu_mining_sentiment', 'source': 'reddit_api', 'keywords': ['GPU mining'], 'smoothing': 'rolling'},
        {'name': 'datacenter_investment', 'source': 'fred', 'keywords': ['data center investment'], 'smoothing': 'EWMA'},
        {'name': 'gaming_market_trends', 'source': 'market_research', 'keywords': ['gaming market'], 'smoothing': 'EWMA'},
        {'name': 'semiconductor_orders', 'source': 'industry_reports', 'keywords': ['semiconductor orders'], 'smoothing': 'EWMA'}
    ],
    'AMD': [
        {'name': 'cpu_benchmarks', 'source': 'tech_news', 'keywords': ['AMD CPU benchmark'], 'smoothing': 'rolling'},
        {'name': 'pc_building_trends', 'source': 'reddit_api', 'keywords': ['PC building'], 'smoothing': 'EWMA'},
        {'name': 'server_market_share', 'source': 'market_research', 'keywords': ['server market'], 'smoothing': 'EWMA'}
    ],
    'MSFT': [
        {'name': 'cloud_adoption_rate', 'source': 'industry_reports', 'keywords': ['cloud adoption'], 'smoothing': 'EWMA'},
        {'name': 'azure_growth', 'source': 'financial_data', 'keywords': ['Azure growth'], 'smoothing': 'EWMA'},
        {'name': 'enterprise_software', 'source': 'fred', 'keywords': ['enterprise software'], 'smoothing': 'EWMA'}
    ]
}

# Volatility adjustment config
VOL_ADJUST_THRESHOLD = 0.03  # Base threshold for buy/sell
CONFIDENCE_MIN = 0.7  # Minimum confidence for action
