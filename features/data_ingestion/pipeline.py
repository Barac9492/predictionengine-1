# features/data_ingestion/pipeline.py
import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from arch import arch_model
import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from shared.config.targets import TARGET_STOCKS, PROXY_IDEAS

def fetch_stock_data(stock, start_date='2021-01-01', end_date=None):
    """Fetch historical stock prices and compute returns."""
    data = yf.download(stock, start=start_date, end=end_date)
    data['returns'] = data['Close'].pct_change()
    return data

def fetch_proxy_data(stock, proxy, start_date='2021-01-01', retries=3):
    """Enhanced proxy data fetching with error handling and multiple sources."""
    for attempt in range(retries):
        try:
            if proxy['source'] == 'pytrends':
                pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=2, backoff_factor=0.1)
                keywords = proxy.get('keywords', [proxy['name'].replace('_', ' ')])
                if isinstance(keywords, str):
                    keywords = [keywords]
                
                # Calculate timeframe from start_date
                start_dt = pd.to_datetime(start_date)
                timeframe = f"{start_dt.strftime('%Y-%m-%d')} {datetime.now().strftime('%Y-%m-%d')}"
                
                pytrends.build_payload(keywords, timeframe=timeframe, geo='US')
                data = pytrends.interest_over_time()
                
                if not data.empty and keywords[0] in data.columns:
                    series = data[keywords[0]]
                    series.name = proxy['name']
                    return series
                
            elif proxy['source'] == 'wikipedia':
                # Wikipedia page views (simplified)
                return fetch_wikipedia_views(proxy['name'], start_date)
                
            elif proxy['source'] == 'fred':
                # Federal Reserve Economic Data
                return fetch_fred_data(proxy['indicator'], start_date)
                
            elif proxy['source'] == 'reddit':
                # Reddit mentions/sentiment (placeholder)
                return fetch_reddit_sentiment(proxy['subreddit'], proxy.get('keywords', [stock]), start_date)
                
            elif proxy['source'] == 'custom_api':
                # Custom API endpoint
                return fetch_custom_api_data(proxy['endpoint'], proxy.get('params', {}), start_date)
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {proxy['name']}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"All attempts failed for {proxy['name']}, returning empty series")
    
    return pd.Series(name=proxy['name'])  # Return empty series on failure

def apply_noise_filter(df, method='EWMA', alpha=0.1, zscore_threshold=3.0):
    """Enhanced noise reduction with outlier detection and multiple smoothing methods."""
    if df.empty:
        return df
    
    # Remove outliers using Z-score
    if len(df) > 3:  # Need minimum data for z-score
        z_scores = np.abs(stats.zscore(df.dropna()))
        df_clean = df[z_scores < zscore_threshold]
        if df_clean.empty:
            df_clean = df  # Fallback if all data removed
    else:
        df_clean = df
    
    # Apply smoothing
    if method == 'EWMA':
        return df_clean.ewm(alpha=alpha).mean()
    elif method == 'rolling':
        window = min(7, len(df_clean))
        return df_clean.rolling(window, min_periods=1).mean()
    elif method == 'savgol':
        from scipy.signal import savgol_filter
        if len(df_clean) >= 5:
            return pd.Series(savgol_filter(df_clean, 5, 2), index=df_clean.index)
    elif method == 'kalman':
        # Simple Kalman-like filtering
        filtered = df_clean.copy()
        if len(filtered) > 1:
            for i in range(1, len(filtered)):
                if pd.notna(filtered.iloc[i]) and pd.notna(filtered.iloc[i-1]):
                    filtered.iloc[i] = 0.8 * filtered.iloc[i-1] + 0.2 * filtered.iloc[i]
        return filtered
    
    return df_clean

def estimate_volatility(returns):
    """Estimate volatility using GARCH."""
    model = arch_model(returns.dropna(), vol='Garch', rescale=False)
    res = model.fit(disp='off')
    return res.conditional_volatility

def build_dataset(stock, start_date='2021-01-01', end_date=None, enable_feature_engineering=True):
    """Enhanced pipeline with feature engineering and signal quality assessment."""
    print(f"Building dataset for {stock}...")
    
    # Fetch price data
    price_data = fetch_stock_data(stock, start_date, end_date)
    if price_data.empty:
        print(f"No price data found for {stock}")
        return pd.DataFrame()
    
    # Fetch proxy data
    proxies = PROXY_IDEAS.get(stock, [])
    proxy_dfs = []
    proxy_quality_scores = {}
    
    for p in proxies:
        print(f"Fetching proxy: {p['name']}")
        proxy_series = fetch_proxy_data(stock, p, start_date)
        
        if not proxy_series.empty:
            # Apply noise filtering
            smoothing_method = p.get('smoothing', 'EWMA').split('(')[0]  # Extract method name
            smoothed = apply_noise_filter(proxy_series, method=smoothing_method)
            
            # Calculate proxy quality score (correlation with price returns)
            quality_score = calculate_proxy_quality(smoothed, price_data['returns'])
            proxy_quality_scores[p['name']] = quality_score
            
            # Only include if quality score is reasonable
            if quality_score > 0.1:  # Minimum correlation threshold
                proxy_dfs.append(smoothed.rename(p['name']))
                print(f"  Added {p['name']} (quality: {quality_score:.3f})")
            else:
                print(f"  Skipped {p['name']} (poor quality: {quality_score:.3f})")
        else:
            print(f"  Failed to fetch {p['name']}")
    
    # Combine data
    if proxy_dfs:
        proxy_df = pd.concat(proxy_dfs, axis=1)
        dataset = price_data.join(proxy_df, how='left')
    else:
        dataset = price_data
        print("Warning: No proxy data available, using price data only")
    
    # Forward fill missing values
    dataset = dataset.fillna(method='ffill')
    
    # Estimate volatility
    dataset['volatility'] = estimate_volatility(dataset['returns'])
    
    # Feature engineering
    if enable_feature_engineering:
        dataset = engineer_features(dataset)
    
    # Store metadata
    dataset.attrs['proxy_quality_scores'] = proxy_quality_scores
    dataset.attrs['stock'] = stock
    dataset.attrs['build_timestamp'] = datetime.now()
    
    print(f"Dataset built: {len(dataset)} rows, {len(dataset.columns)} features")
    return dataset.fillna(0)

def fetch_wikipedia_views(page_name, start_date):
    """Fetch Wikipedia page views (simplified placeholder)."""
    # This would use the Wikimedia API in practice
    dates = pd.date_range(start_date, periods=100, freq='D')
    views = np.random.randint(1000, 10000, len(dates))  # Placeholder data
    return pd.Series(views, index=dates, name=f'wiki_{page_name}')

def fetch_fred_data(indicator, start_date):
    """Fetch Federal Reserve Economic Data (placeholder)."""
    # Would use FRED API in practice
    dates = pd.date_range(start_date, periods=50, freq='W')
    data = np.random.randn(len(dates)).cumsum() + 100
    return pd.Series(data, index=dates, name=f'fred_{indicator}')

def fetch_reddit_sentiment(subreddit, keywords, start_date):
    """Fetch Reddit sentiment data (placeholder)."""
    dates = pd.date_range(start_date, periods=100, freq='D')
    sentiment = np.random.randn(len(dates)) * 0.5
    return pd.Series(sentiment, index=dates, name=f'reddit_{subreddit}')

def fetch_custom_api_data(endpoint, params, start_date):
    """Fetch data from custom API endpoint (placeholder)."""
    dates = pd.date_range(start_date, periods=100, freq='D')
    data = np.random.randn(len(dates)).cumsum()
    return pd.Series(data, index=dates, name='custom_api')

def calculate_proxy_quality(proxy_series, price_returns):
    """Calculate correlation-based quality score for proxy data."""
    if proxy_series.empty or price_returns.empty:
        return 0.0
    
    # Align series by date
    aligned = pd.concat([proxy_series, price_returns], axis=1, join='inner')
    if len(aligned) < 10:  # Need minimum data points
        return 0.0
    
    # Calculate correlation
    correlation = aligned.corr().iloc[0, 1]
    if pd.isna(correlation):
        return 0.0
    
    # Return absolute correlation as quality score
    return abs(correlation)

def engineer_features(dataset):
    """Add technical indicators and derived features."""
    # Price-based features
    if 'Close' in dataset.columns:
        # Moving averages
        dataset['ma_5'] = dataset['Close'].rolling(5).mean()
        dataset['ma_20'] = dataset['Close'].rolling(20).mean()
        dataset['ma_ratio'] = dataset['ma_5'] / dataset['ma_20']
        
        # RSI (simplified)
        delta = dataset['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        dataset['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_window = 20
        bb_std = dataset['Close'].rolling(bb_window).std()
        bb_mean = dataset['Close'].rolling(bb_window).mean()
        dataset['bb_upper'] = bb_mean + (bb_std * 2)
        dataset['bb_lower'] = bb_mean - (bb_std * 2)
        dataset['bb_position'] = (dataset['Close'] - bb_lower) / (bb_upper - bb_lower)
    
    # Volume features
    if 'Volume' in dataset.columns:
        dataset['volume_ma'] = dataset['Volume'].rolling(20).mean()
        dataset['volume_ratio'] = dataset['Volume'] / dataset['volume_ma']
    
    # Proxy momentum features
    proxy_cols = [col for col in dataset.columns if col not in 
                 ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'returns', 'volatility']]
    
    for col in proxy_cols:
        if col in dataset.columns:
            # Momentum (rate of change)
            dataset[f'{col}_momentum'] = dataset[col].pct_change(5)
            # Z-score (standardized value)
            dataset[f'{col}_zscore'] = (dataset[col] - dataset[col].rolling(30).mean()) / dataset[col].rolling(30).std()
    
    return dataset

# Example usage
if __name__ == '__main__':
    df = build_dataset('TSLA')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    if hasattr(df, 'attrs') and 'proxy_quality_scores' in df.attrs:
        print(f"Proxy quality scores: {df.attrs['proxy_quality_scores']}")
    print(df.head())
