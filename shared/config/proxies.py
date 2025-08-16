# shared/config/proxies.py
# Expanded proxy ideas for each stock, focusing on noise-resilient public data sources

from .targets import TARGET_STOCKS

PROXY_IDEAS = {
    'TSLA': [
        {'name': 'google_trends_ev', 'source': 'pytrends', 'keywords': ['electric vehicle'], 'smoothing': 'EWMA (alpha=0.1)'},
        {'name': 'job_postings_auto', 'source': 'BLS API', 'category': 'automotive', 'smoothing': 'Z-score outlier removal + rolling 30d'},
        {'name': 'wikipedia_views_tesla', 'source': 'Wikimedia API', 'page': 'Tesla, Inc.', 'smoothing': 'rolling average 7d'},
        {'name': 'patent_filings_ev', 'source': 'USPTO API', 'query': 'electric vehicle', 'smoothing': 'monthly aggregate'},
    ],
    'AAPL': [
        {'name': 'google_trends_iphone', 'source': 'pytrends', 'keywords': ['iPhone'], 'smoothing': 'EWMA (alpha=0.05)'},
        {'name': 'app_store_rankings', 'source': 'App Annie API (free tier)', 'app': 'Apple apps', 'smoothing': 'weekly average'},
        {'name': 'wikipedia_views_apple', 'source': 'Wikimedia API', 'page': 'Apple Inc.', 'smoothing': 'rolling 7d'},
        {'name': 'job_postings_tech', 'source': 'Indeed API', 'query': 'Apple jobs', 'smoothing': 'EWMA'},
    ],
    'NVDA': [
        {'name': 'google_trends_gpu', 'source': 'pytrends', 'keywords': ['GPU', 'Nvidia'], 'smoothing': 'EWMA'},
        {'name': 'crypto_mining_trends', 'source': 'pytrends', 'keywords': ['crypto mining'], 'smoothing': 'rolling 14d'},
        {'name': 'wikipedia_views_nvidia', 'source': 'Wikimedia API', 'page': 'Nvidia', 'smoothing': 'rolling 7d'},
    ],
    'AMD': [
        {'name': 'google_trends_cpu', 'source': 'pytrends', 'keywords': ['AMD CPU'], 'smoothing': 'EWMA'},
        {'name': 'gaming_forums_activity', 'source': 'Reddit API', 'subreddit': 'amd', 'smoothing': 'daily count smoothed'},
    ],
    'MSFT': [
        {'name': 'google_trends_azure', 'source': 'pytrends', 'keywords': ['Azure cloud'], 'smoothing': 'EWMA'},
        {'name': 'github_stars_microsoft', 'source': 'GitHub API', 'org': 'microsoft', 'smoothing': 'weekly growth'},
    ],
    'GOOGL': [
        {'name': 'google_trends_android', 'source': 'pytrends', 'keywords': ['Android'], 'smoothing': 'EWMA'},
        {'name': 'search_volume_google', 'source': 'pytrends', 'keywords': ['Google search'], 'smoothing': 'rolling 7d'},
    ],
    'AMZN': [
        {'name': 'google_trends_amazon', 'source': 'pytrends', 'keywords': ['Amazon shopping'], 'smoothing': 'EWMA'},
        {'name': 'job_postings_logistics', 'source': 'BLS API', 'category': 'ecommerce', 'smoothing': 'monthly'},
    ],
    'META': [
        {'name': 'google_trends_facebook', 'source': 'pytrends', 'keywords': ['Facebook'], 'smoothing': 'EWMA'},
        {'name': 'app_downloads_meta', 'source': 'SensorTower', 'app': 'Facebook', 'smoothing': 'weekly'},
    ],
    'NFLX': [
        {'name': 'google_trends_netflix', 'source': 'pytrends', 'keywords': ['Netflix'], 'smoothing': 'EWMA'},
        {'name': 'imdb_ratings_new', 'source': 'IMDB API', 'query': 'new releases', 'smoothing': 'rolling 30d'},
    ],
    'COIN': [
        {'name': 'google_trends_crypto', 'source': 'pytrends', 'keywords': ['cryptocurrency'], 'smoothing': 'EWMA (alpha=0.2)'},
        {'name': 'blockchain_transactions', 'source': 'Blockchain API', 'chain': 'bitcoin', 'smoothing': 'daily average'},
    ],
}

# Function to get proxies for a stock
def get_proxies(stock):
    return PROXY_IDEAS.get(stock, [])
