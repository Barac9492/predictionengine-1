#!/usr/bin/env python3
"""
Run predictions and backtesting for key stocks: AAPL, GOOGL, NVDA, TSLA
Shows what each prediction is based on and demonstrates prediction power.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from features.data_ingestion.pipeline import build_dataset
from features.modeling.predictor import EnsemblePredictor
from features.trading_decision.guidance import TradingGuidanceEngine
from features.proxy_discovery.discovery import ProxyDiscoveryEngine
from tests.backtesting_framework import BacktestingFramework

# Target stocks
TARGET_STOCKS = ['AAPL', 'GOOGL', 'NVDA', 'TSLA']

class StockPredictionAnalyzer:
    """
    Comprehensive analyzer that shows:
    1. What each prediction is based on (features, proxies)
    2. Backtesting results to prove prediction power
    3. Clear metrics and visualizations
    """
    
    def __init__(self):
        self.guidance_engine = TradingGuidanceEngine()
        self.proxy_discovery = ProxyDiscoveryEngine()
        self.backtester = BacktestingFramework(initial_capital=10000)
        self.results = {}
        
    def analyze_stock(self, stock: str, backtest_period: int = 365) -> Dict:
        """
        Complete analysis for a single stock.
        """
        print(f"\n{'='*60}")
        print(f"📊 Analyzing {stock}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Build dataset with features and proxies
            print(f"📥 Loading data and features for {stock}...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=backtest_period + 100)  # Extra data for training
            
            data = build_dataset(stock, start_date=start_date.strftime('%Y-%m-%d'))
            
            if data.empty:
                print(f"❌ No data available for {stock}")
                return None
            
            print(f"✅ Loaded {len(data)} days of data with {len(data.columns)} features")
            
            # Step 2: Identify key features and proxies
            features_analysis = self.analyze_features(data, stock)
            
            # Step 3: Make current prediction
            current_prediction = self.make_prediction(data, stock)
            
            # Step 4: Run backtesting to prove prediction power
            backtest_results = self.run_backtest(stock, data)
            
            # Step 5: Compile comprehensive results
            result = {
                'stock': stock,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(data),
                'features_analysis': features_analysis,
                'current_prediction': current_prediction,
                'backtest_results': backtest_results,
                'prediction_drivers': self.identify_prediction_drivers(data, current_prediction)
            }
            
            self.results[stock] = result
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing {stock}: {str(e)}")
            return None
    
    def analyze_features(self, data: pd.DataFrame, stock: str) -> Dict:
        """
        Analyze which features are most important for predictions.
        """
        print(f"🔍 Analyzing features and proxies...")
        
        # Identify different types of features
        price_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'returns', 'volatility']
        technical_features = [col for col in data.columns if any(
            indicator in col for indicator in ['ma_', 'rsi', 'bb_', 'volume_ratio']
        )]
        proxy_features = [col for col in data.columns if col not in price_features + technical_features]
        
        # Calculate feature importance (correlation with returns)
        feature_importance = {}
        if 'returns' in data.columns:
            for col in data.columns:
                if col != 'returns' and not data[col].isna().all():
                    # Calculate correlation
                    try:
                        corr = data[col].corr(data['returns'])
                        if not pd.isna(corr):
                            feature_importance[col] = abs(corr)
                    except:
                        feature_importance[col] = 0.0
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        analysis = {
            'total_features': len(data.columns),
            'price_features': len([f for f in price_features if f in data.columns]),
            'technical_indicators': len([f for f in technical_features if f in data.columns]),
            'proxy_indicators': len([f for f in proxy_features if f in data.columns]),
            'top_5_important_features': sorted_features[:5] if sorted_features else [],
            'feature_categories': {
                'price_based': [f for f in price_features if f in data.columns][:5],
                'technical': technical_features[:5],
                'proxies': proxy_features[:5]
            }
        }
        
        # Print summary
        print(f"  📈 Price features: {analysis['price_features']}")
        print(f"  📊 Technical indicators: {analysis['technical_indicators']}")
        print(f"  🔗 Proxy indicators: {analysis['proxy_indicators']}")
        
        if analysis['top_5_important_features']:
            print(f"\n  🎯 Top 5 Most Important Features (by correlation):")
            for i, (feature, importance) in enumerate(analysis['top_5_important_features'], 1):
                print(f"    {i}. {feature}: {importance:.3f}")
        
        return analysis
    
    def make_prediction(self, data: pd.DataFrame, stock: str) -> Dict:
        """
        Make current prediction and explain what it's based on.
        """
        print(f"\n🤖 Generating prediction for {stock}...")
        
        try:
            # Train predictor on historical data
            train_data = data[:-1]  # Keep last row for prediction
            predictor = EnsemblePredictor(train_data, target_col='returns')
            
            # Make prediction for next period
            recent_data = data.iloc[-30:]  # Use last 30 days
            pred_mean, pred_std = predictor.predict(recent_data)
            
            # Calculate confidence and get guidance
            confidence = 1 / (1 + pred_std) if pred_std > 0 else 0.5
            volatility = data['volatility'].iloc[-1] if 'volatility' in data else 0.02
            
            action, rationale, metrics = self.guidance_engine.get_guidance(
                pred_mean, confidence, volatility
            )
            
            # Get current price and calculate target
            current_price = data['Close'].iloc[-1]
            predicted_price = current_price * (1 + pred_mean)
            price_change_dollars = predicted_price - current_price
            
            prediction = {
                'action': action,
                'rationale': rationale,
                'current_price': round(current_price, 2),
                'predicted_change_percent': round(pred_mean * 100, 2),
                'predicted_price': round(predicted_price, 2),
                'price_change_dollars': round(price_change_dollars, 2),
                'confidence': round(confidence * 100, 1),
                'volatility': round(volatility * 100, 2),
                'prediction_horizon': '1-5 days',
                'metrics': metrics
            }
            
            # Print prediction
            print(f"  💰 Current Price: ${prediction['current_price']}")
            print(f"  🎯 Prediction: {action}")
            print(f"  📈 Expected Change: {prediction['predicted_change_percent']:+.2f}%")
            print(f"  💵 Target Price: ${prediction['predicted_price']}")
            print(f"  🎲 Confidence: {prediction['confidence']:.1f}%")
            print(f"  📊 Volatility: {prediction['volatility']:.2f}%")
            print(f"  💡 Rationale: {rationale}")
            
            return prediction
            
        except Exception as e:
            print(f"  ❌ Prediction failed: {str(e)}")
            return {
                'action': 'Hold',
                'error': str(e),
                'rationale': 'Unable to generate prediction due to error'
            }
    
    def run_backtest(self, stock: str, data: pd.DataFrame) -> Dict:
        """
        Run backtesting to demonstrate prediction power.
        """
        print(f"\n📊 Running backtest for {stock}...")
        
        try:
            # Run backtest for last year
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            result = self.backtester.run_backtest(
                stock=stock,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                prediction_window=20,
                rebalance_freq='weekly'
            )
            
            # Calculate additional metrics
            buy_and_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[-365] - 1) if len(data) > 365 else 0
            outperformance = result.total_return - buy_and_hold_return
            
            backtest_summary = {
                'period': '1 Year',
                'total_return': round(result.total_return * 100, 2),
                'annualized_return': round(result.annualized_return * 100, 2),
                'sharpe_ratio': round(result.sharpe_ratio, 3),
                'max_drawdown': round(result.max_drawdown * 100, 2),
                'win_rate': round(result.win_rate * 100, 1),
                'total_trades': result.total_trades,
                'profit_factor': round(result.profit_factor, 2),
                'buy_and_hold_return': round(buy_and_hold_return * 100, 2),
                'outperformance': round(outperformance * 100, 2),
                'prediction_accuracy': round(result.win_rate * 100, 1) if result.win_rate > 0 else 0
            }
            
            # Print backtest results
            print(f"  📈 Strategy Return: {backtest_summary['total_return']:+.2f}%")
            print(f"  📊 Buy & Hold Return: {backtest_summary['buy_and_hold_return']:+.2f}%")
            print(f"  🎯 Outperformance: {backtest_summary['outperformance']:+.2f}%")
            print(f"  📉 Max Drawdown: {backtest_summary['max_drawdown']:.2f}%")
            print(f"  ✅ Win Rate: {backtest_summary['win_rate']:.1f}%")
            print(f"  📊 Sharpe Ratio: {backtest_summary['sharpe_ratio']:.3f}")
            print(f"  🔄 Total Trades: {backtest_summary['total_trades']}")
            
            return backtest_summary
            
        except Exception as e:
            print(f"  ❌ Backtest failed: {str(e)}")
            return {'error': str(e)}
    
    def identify_prediction_drivers(self, data: pd.DataFrame, prediction: Dict) -> Dict:
        """
        Identify what's driving the current prediction.
        """
        drivers = {
            'primary_drivers': [],
            'technical_signals': [],
            'proxy_signals': [],
            'market_conditions': []
        }
        
        # Check recent price momentum
        if len(data) > 5:
            recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-5] - 1)
            if abs(recent_return) > 0.02:
                drivers['primary_drivers'].append({
                    'factor': 'Price Momentum',
                    'value': f"{recent_return*100:+.2f}% (5-day)",
                    'impact': 'High' if abs(recent_return) > 0.05 else 'Medium'
                })
        
        # Check technical indicators
        if 'rsi' in data.columns and not pd.isna(data['rsi'].iloc[-1]):
            rsi_value = data['rsi'].iloc[-1]
            if rsi_value > 70:
                drivers['technical_signals'].append({
                    'indicator': 'RSI',
                    'value': f"{rsi_value:.1f}",
                    'signal': 'Overbought'
                })
            elif rsi_value < 30:
                drivers['technical_signals'].append({
                    'indicator': 'RSI',
                    'value': f"{rsi_value:.1f}",
                    'signal': 'Oversold'
                })
        
        # Check moving averages
        if 'ma_ratio' in data.columns and not pd.isna(data['ma_ratio'].iloc[-1]):
            ma_ratio = data['ma_ratio'].iloc[-1]
            if ma_ratio > 1.02:
                drivers['technical_signals'].append({
                    'indicator': 'MA Crossover',
                    'value': f"{ma_ratio:.3f}",
                    'signal': 'Bullish'
                })
            elif ma_ratio < 0.98:
                drivers['technical_signals'].append({
                    'indicator': 'MA Crossover',
                    'value': f"{ma_ratio:.3f}",
                    'signal': 'Bearish'
                })
        
        # Check volatility conditions
        if 'volatility' in data.columns:
            current_vol = data['volatility'].iloc[-1]
            avg_vol = data['volatility'].mean()
            if current_vol > avg_vol * 1.5:
                drivers['market_conditions'].append({
                    'condition': 'High Volatility',
                    'current': f"{current_vol*100:.2f}%",
                    'average': f"{avg_vol*100:.2f}%"
                })
        
        # Add proxy signals if available
        proxy_cols = [col for col in data.columns if 'google_trends' in col or 'sentiment' in col]
        for col in proxy_cols[:3]:  # Top 3 proxy signals
            if not pd.isna(data[col].iloc[-1]):
                recent_change = (data[col].iloc[-1] / data[col].iloc[-5] - 1) if len(data) > 5 else 0
                if abs(recent_change) > 0.1:
                    drivers['proxy_signals'].append({
                        'proxy': col.replace('_', ' ').title(),
                        'change': f"{recent_change*100:+.2f}% (5-day)",
                        'strength': 'Strong' if abs(recent_change) > 0.2 else 'Moderate'
                    })
        
        return drivers
    
    def generate_report(self) -> str:
        """
        Generate comprehensive report for all stocks.
        """
        report = []
        report.append("\n" + "="*80)
        report.append("📊 STOCK PREDICTION & BACKTESTING REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for stock, result in self.results.items():
            if result:
                report.append(f"\n{'='*60}")
                report.append(f"🎯 {stock} - ANALYSIS SUMMARY")
                report.append(f"{'='*60}")
                
                # Current Prediction
                pred = result['current_prediction']
                report.append(f"\n📈 CURRENT PREDICTION:")
                report.append(f"  Action: {pred['action']}")
                report.append(f"  Current Price: ${pred.get('current_price', 'N/A')}")
                report.append(f"  Expected Change: {pred.get('predicted_change_percent', 0):+.2f}%")
                report.append(f"  Target Price: ${pred.get('predicted_price', 'N/A')}")
                report.append(f"  Confidence: {pred.get('confidence', 0):.1f}%")
                
                # What It's Based On
                report.append(f"\n🔍 PREDICTION BASED ON:")
                features = result['features_analysis']
                report.append(f"  Total Features: {features['total_features']}")
                report.append(f"  - Price Features: {features['price_features']}")
                report.append(f"  - Technical Indicators: {features['technical_indicators']}")
                report.append(f"  - Proxy Indicators: {features['proxy_indicators']}")
                
                if features['top_5_important_features']:
                    report.append(f"\n  Top Predictive Features:")
                    for i, (feature, importance) in enumerate(features['top_5_important_features'][:5], 1):
                        report.append(f"    {i}. {feature}: {importance:.3f} correlation")
                
                # Prediction Drivers
                drivers = result['prediction_drivers']
                if drivers['primary_drivers']:
                    report.append(f"\n  Primary Drivers:")
                    for driver in drivers['primary_drivers']:
                        report.append(f"    • {driver['factor']}: {driver['value']}")
                
                # Backtesting Results
                backtest = result['backtest_results']
                if 'total_return' in backtest:
                    report.append(f"\n📊 BACKTESTING RESULTS (1 Year):")
                    report.append(f"  Strategy Return: {backtest['total_return']:+.2f}%")
                    report.append(f"  Buy & Hold Return: {backtest['buy_and_hold_return']:+.2f}%")
                    report.append(f"  Outperformance: {backtest['outperformance']:+.2f}%")
                    report.append(f"  Win Rate: {backtest['win_rate']:.1f}%")
                    report.append(f"  Sharpe Ratio: {backtest['sharpe_ratio']:.3f}")
                    report.append(f"  Max Drawdown: {backtest['max_drawdown']:.2f}%")
                    report.append(f"  Total Trades: {backtest['total_trades']}")
                    
                    # Performance Rating
                    if backtest['outperformance'] > 10:
                        rating = "⭐⭐⭐⭐⭐ Excellent"
                    elif backtest['outperformance'] > 5:
                        rating = "⭐⭐⭐⭐ Very Good"
                    elif backtest['outperformance'] > 0:
                        rating = "⭐⭐⭐ Good"
                    else:
                        rating = "⭐⭐ Needs Improvement"
                    report.append(f"  Performance Rating: {rating}")
        
        report.append(f"\n{'='*80}")
        report.append("📝 SUMMARY")
        report.append(f"{'='*80}")
        
        # Overall statistics
        successful_stocks = [s for s, r in self.results.items() if r and 'backtest_results' in r]
        if successful_stocks:
            avg_return = np.mean([self.results[s]['backtest_results']['total_return'] 
                                 for s in successful_stocks if 'total_return' in self.results[s]['backtest_results']])
            avg_outperformance = np.mean([self.results[s]['backtest_results']['outperformance'] 
                                         for s in successful_stocks if 'outperformance' in self.results[s]['backtest_results']])
            
            report.append(f"Stocks Analyzed: {len(successful_stocks)}/{len(TARGET_STOCKS)}")
            report.append(f"Average Strategy Return: {avg_return:+.2f}%")
            report.append(f"Average Outperformance: {avg_outperformance:+.2f}%")
            
            # Current recommendations
            report.append(f"\n🎯 CURRENT RECOMMENDATIONS:")
            for stock in successful_stocks:
                pred = self.results[stock]['current_prediction']
                report.append(f"  {stock}: {pred['action']} (Confidence: {pred.get('confidence', 0):.1f}%)")
        
        report.append(f"\n⚠️ DISCLAIMER: This is for educational purposes only. Not financial advice.")
        report.append("="*80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """
        Save results to JSON file.
        """
        if not filename:
            filename = f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert results to JSON-serializable format
        json_results = {}
        for stock, result in self.results.items():
            if result:
                json_results[stock] = {
                    'timestamp': result['timestamp'],
                    'current_prediction': result['current_prediction'],
                    'backtest_results': result['backtest_results'],
                    'features_summary': {
                        'total_features': result['features_analysis']['total_features'],
                        'top_features': [
                            {'name': f[0], 'importance': float(f[1])} 
                            for f in result['features_analysis']['top_5_important_features']
                        ]
                    }
                }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename

def main():
    """
    Main function to run predictions and backtesting for all target stocks.
    """
    print("\n" + "="*80)
    print("🚀 GENIUS PREDICTION ENGINE - STOCK ANALYSIS")
    print("="*80)
    print(f"Analyzing: {', '.join(TARGET_STOCKS)}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize analyzer
    analyzer = StockPredictionAnalyzer()
    
    # Analyze each stock
    for stock in TARGET_STOCKS:
        analyzer.analyze_stock(stock)
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Save results
    filename = analyzer.save_results()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print(f"Results saved to: {filename}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()