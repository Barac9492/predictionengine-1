# features/proxy_discovery/discovery.py
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import time
from datetime import datetime, timedelta
import re
from collections import defaultdict
from scipy.stats import pearsonr
from shared.config.targets import TARGET_STOCKS

@dataclass
class ProxyCandidate:
    """Data class for proxy candidate information."""
    name: str
    description: str
    source: str
    keywords: List[str]
    correlation_score: float = 0.0
    signal_to_noise_ratio: float = 0.0
    data_availability: float = 0.0  # Percentage of data points available
    latency_score: float = 0.0  # How quickly data becomes available
    discovery_method: str = ""
    confidence: float = 0.0

class ProxyDiscoveryEngine:
    """
    Automated proxy discovery system that finds indirect indicators for stock prediction.
    Uses various sources and heuristics to identify potential predictive relationships.
    """
    
    def __init__(self):
        self.discovered_proxies = {}
        self.proxy_performance_history = defaultdict(list)
        self.discovery_strategies = [
            'business_model_analysis',
            'supply_chain_analysis', 
            'competitor_analysis',
            'news_trend_analysis',
            'social_sentiment_analysis',
            'economic_indicator_analysis'
        ]
        
    def discover_proxies_for_stock(self, stock: str, company_info: Dict = None) -> List[ProxyCandidate]:
        """
        Main discovery method that finds proxy candidates for a given stock.
        """
        print(f"Discovering proxies for {stock}...")
        
        all_candidates = []
        
        # Get company information if not provided
        if not company_info:
            company_info = self.get_company_info(stock)
        
        # Apply different discovery strategies
        for strategy in self.discovery_strategies:
            candidates = getattr(self, f'_strategy_{strategy}')(stock, company_info)
            all_candidates.extend(candidates)
            print(f"  {strategy}: found {len(candidates)} candidates")
        
        # Remove duplicates and rank candidates
        unique_candidates = self.deduplicate_candidates(all_candidates)
        ranked_candidates = self.rank_candidates(unique_candidates, stock)
        
        print(f"Total unique candidates: {len(unique_candidates)}")
        print(f"Top 10 ranked candidates:")
        for i, candidate in enumerate(ranked_candidates[:10]):
            print(f"  {i+1}. {candidate.name} (conf: {candidate.confidence:.3f})")
        
        return ranked_candidates
    
    def get_company_info(self, stock: str) -> Dict:
        """Get basic company information for proxy discovery."""
        # This would integrate with APIs like Yahoo Finance, SEC filings, etc.
        # For now, using a predefined mapping
        company_data = {
            'TSLA': {
                'name': 'Tesla Inc',
                'sector': 'Automotive',
                'industry': 'Electric Vehicles',
                'keywords': ['electric vehicle', 'EV', 'battery', 'autonomous driving', 'clean energy'],
                'supply_chain': ['lithium', 'cobalt', 'semiconductors'],
                'competitors': ['NIO', 'RIVN', 'LCID'],
                'business_model': 'manufacturing_and_software'
            },
            'AAPL': {
                'name': 'Apple Inc',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'keywords': ['iPhone', 'smartphone', 'consumer electronics', 'app store'],
                'supply_chain': ['semiconductors', 'rare earth metals'],
                'competitors': ['GOOGL', 'MSFT', 'AMZN'],
                'business_model': 'hardware_and_services'
            },
            'NVDA': {
                'name': 'NVIDIA Corporation',
                'sector': 'Technology',
                'industry': 'Semiconductors',
                'keywords': ['GPU', 'AI', 'machine learning', 'gaming', 'data center'],
                'supply_chain': ['semiconductor fabrication', 'TSMC'],
                'competitors': ['AMD', 'INTC'],
                'business_model': 'chip_design'
            }
        }
        return company_data.get(stock, {'name': stock, 'keywords': [stock.lower()]})
    
    def _strategy_business_model_analysis(self, stock: str, company_info: Dict) -> List[ProxyCandidate]:
        """Discover proxies based on business model analysis."""
        candidates = []
        
        if company_info.get('business_model') == 'manufacturing_and_software':
            # For companies like Tesla
            candidates.extend([
                ProxyCandidate(
                    name='global_ev_sales',
                    description='Global electric vehicle sales data',
                    source='industry_reports',
                    keywords=['electric vehicle sales', 'EV market'],
                    discovery_method='business_model_analysis'
                ),
                ProxyCandidate(
                    name='battery_price_index',
                    description='Lithium battery price trends',
                    source='commodity_data',
                    keywords=['lithium battery prices', 'battery cost'],
                    discovery_method='business_model_analysis'
                )
            ])
        
        if company_info.get('business_model') == 'hardware_and_services':
            # For companies like Apple
            candidates.extend([
                ProxyCandidate(
                    name='smartphone_market_share',
                    description='Global smartphone market trends',
                    source='market_research',
                    keywords=['smartphone sales', 'mobile market'],
                    discovery_method='business_model_analysis'
                ),
                ProxyCandidate(
                    name='app_store_revenue',
                    description='Mobile app store revenue trends',
                    source='app_analytics',
                    keywords=['app store revenue', 'mobile apps'],
                    discovery_method='business_model_analysis'
                )
            ])
        
        return candidates
    
    def _strategy_supply_chain_analysis(self, stock: str, company_info: Dict) -> List[ProxyCandidate]:
        """Discover proxies based on supply chain dependencies."""
        candidates = []
        supply_chain = company_info.get('supply_chain', [])
        
        for component in supply_chain:
            candidates.append(ProxyCandidate(
                name=f'{component}_price_index',
                description=f'Price trends for {component}',
                source='commodity_data',
                keywords=[component, f'{component} price'],
                discovery_method='supply_chain_analysis'
            ))
            
            candidates.append(ProxyCandidate(
                name=f'{component}_supply_news',
                description=f'News sentiment about {component} supply',
                source='news_sentiment',
                keywords=[f'{component} supply', f'{component} shortage'],
                discovery_method='supply_chain_analysis'
            ))
        
        return candidates
    
    def _strategy_competitor_analysis(self, stock: str, company_info: Dict) -> List[ProxyCandidate]:
        """Discover proxies based on competitor performance."""
        candidates = []
        competitors = company_info.get('competitors', [])
        
        for competitor in competitors:
            candidates.append(ProxyCandidate(
                name=f'{competitor}_performance_ratio',
                description=f'Performance ratio vs {competitor}',
                source='financial_data',
                keywords=[competitor],
                discovery_method='competitor_analysis'
            ))
        
        # Industry sector performance
        sector = company_info.get('sector', '')
        if sector:
            candidates.append(ProxyCandidate(
                name=f'{sector.lower()}_sector_etf',
                description=f'{sector} sector ETF performance',
                source='financial_data',
                keywords=[f'{sector} ETF', f'{sector} sector'],
                discovery_method='competitor_analysis'
            ))
        
        return candidates
    
    def _strategy_news_trend_analysis(self, stock: str, company_info: Dict) -> List[ProxyCandidate]:
        """Discover proxies based on news and trend analysis."""
        candidates = []
        keywords = company_info.get('keywords', [])
        
        for keyword in keywords:
            candidates.extend([
                ProxyCandidate(
                    name=f'{keyword.replace(" ", "_")}_google_trends',
                    description=f'Google Trends for "{keyword}"',
                    source='pytrends',
                    keywords=[keyword],
                    discovery_method='news_trend_analysis'
                ),
                ProxyCandidate(
                    name=f'{keyword.replace(" ", "_")}_news_sentiment',
                    description=f'News sentiment for "{keyword}"',
                    source='news_sentiment',
                    keywords=[keyword],
                    discovery_method='news_trend_analysis'
                )
            ])
        
        return candidates
    
    def _strategy_social_sentiment_analysis(self, stock: str, company_info: Dict) -> List[ProxyCandidate]:
        """Discover proxies based on social media sentiment."""
        candidates = []
        
        # Twitter/X mentions
        candidates.append(ProxyCandidate(
            name=f'{stock}_twitter_sentiment',
            description=f'Twitter sentiment for ${stock}',
            source='twitter_api',
            keywords=[f'${stock}', company_info.get('name', stock)],
            discovery_method='social_sentiment_analysis'
        ))
        
        # Reddit mentions
        relevant_subreddits = ['investing', 'stocks', 'wallstreetbets', 'SecurityAnalysis']
        for subreddit in relevant_subreddits:
            candidates.append(ProxyCandidate(
                name=f'{stock}_{subreddit}_mentions',
                description=f'Reddit mentions in r/{subreddit}',
                source='reddit_api',
                keywords=[stock, company_info.get('name', stock)],
                discovery_method='social_sentiment_analysis'
            ))
        
        return candidates
    
    def _strategy_economic_indicator_analysis(self, stock: str, company_info: Dict) -> List[ProxyCandidate]:
        """Discover proxies based on economic indicators."""
        candidates = []
        sector = company_info.get('sector', '')
        
        # General economic indicators
        general_indicators = [
            ('consumer_confidence', 'Consumer Confidence Index'),
            ('unemployment_rate', 'Unemployment Rate'),
            ('gdp_growth', 'GDP Growth Rate'),
            ('inflation_rate', 'Inflation Rate')
        ]
        
        for indicator_id, description in general_indicators:
            candidates.append(ProxyCandidate(
                name=f'{indicator_id}_{stock}',
                description=f'{description} impact on {stock}',
                source='fred',
                keywords=[indicator_id.replace('_', ' ')],
                discovery_method='economic_indicator_analysis'
            ))
        
        # Sector-specific indicators
        if sector == 'Technology':
            candidates.append(ProxyCandidate(
                name='tech_spending_index',
                description='Corporate technology spending index',
                source='fred',
                keywords=['technology spending', 'IT investment'],
                discovery_method='economic_indicator_analysis'
            ))
        elif sector == 'Automotive':
            candidates.append(ProxyCandidate(
                name='auto_sales_rate',
                description='Vehicle sales rate',
                source='fred',
                keywords=['auto sales', 'vehicle sales'],
                discovery_method='economic_indicator_analysis'
            ))
        
        return candidates
    
    def deduplicate_candidates(self, candidates: List[ProxyCandidate]) -> List[ProxyCandidate]:
        """Remove duplicate candidates based on name and keywords similarity."""
        unique_candidates = []
        seen_names = set()
        
        for candidate in candidates:
            # Simple deduplication by name
            if candidate.name not in seen_names:
                unique_candidates.append(candidate)
                seen_names.add(candidate.name)
        
        return unique_candidates
    
    def rank_candidates(self, candidates: List[ProxyCandidate], stock: str) -> List[ProxyCandidate]:
        """Rank candidates by potential predictive value."""
        for candidate in candidates:
            # Calculate composite confidence score
            # This is a simplified scoring system - in practice would use ML
            
            strategy_weights = {
                'business_model_analysis': 0.3,
                'supply_chain_analysis': 0.25,
                'competitor_analysis': 0.15,
                'news_trend_analysis': 0.1,
                'social_sentiment_analysis': 0.1,
                'economic_indicator_analysis': 0.1
            }
            
            base_score = strategy_weights.get(candidate.discovery_method, 0.1)
            
            # Adjust based on data source reliability
            source_weights = {
                'pytrends': 0.8,
                'financial_data': 0.9,
                'commodity_data': 0.85,
                'news_sentiment': 0.7,
                'twitter_api': 0.6,
                'reddit_api': 0.5,
                'fred': 0.95
            }
            
            source_score = source_weights.get(candidate.source, 0.5)
            
            # Calculate final confidence
            candidate.confidence = base_score * source_score
        
        # Sort by confidence score
        return sorted(candidates, key=lambda x: x.confidence, reverse=True)
    
    def evaluate_proxy_performance(self, proxy_name: str, stock: str, 
                                 correlation_score: float, signal_to_noise: float) -> None:
        """Update proxy performance tracking."""
        self.proxy_performance_history[f"{stock}_{proxy_name}"].append({
            'timestamp': datetime.now(),
            'correlation': correlation_score,
            'signal_to_noise': signal_to_noise
        })
    
    def get_top_proxies_for_stock(self, stock: str, limit: int = 5) -> List[Dict]:
        """Get the top performing proxies for a stock."""
        candidates = self.discover_proxies_for_stock(stock)
        top_candidates = candidates[:limit]
        
        # Convert to dict format for easy integration
        return [asdict(candidate) for candidate in top_candidates]
    
    def export_proxy_config(self, stock: str, filename: str = None) -> str:
        """Export discovered proxies to configuration format."""
        proxies = self.get_top_proxies_for_stock(stock)
        
        config = {
            stock: []
        }
        
        for proxy in proxies:
            config[stock].append({
                'name': proxy['name'],
                'source': proxy['source'],
                'keywords': proxy['keywords'],
                'smoothing': 'EWMA',
                'confidence': proxy['confidence']
            })
        
        if not filename:
            filename = f'discovered_proxies_{stock}_{datetime.now().strftime("%Y%m%d")}.json'
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        return filename

# Usage example and testing
if __name__ == '__main__':
    discovery_engine = ProxyDiscoveryEngine()
    
    # Test discovery for Tesla
    tesla_proxies = discovery_engine.discover_proxies_for_stock('TSLA')
    print(f"\nTop 5 proxies for TSLA:")
    for i, proxy in enumerate(tesla_proxies[:5]):
        print(f"{i+1}. {proxy.name} - {proxy.description} (confidence: {proxy.confidence:.3f})")
    
    # Export configuration
    config_file = discovery_engine.export_proxy_config('TSLA')
    print(f"\nExported configuration to: {config_file}")