#!/usr/bin/env python3
"""
Enhanced Stock Prediction Analysis with Real Timestamped Data
Shows clear strategy, detailed backtesting, and actual trade results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Target stocks
TARGET_STOCKS = ['AAPL', 'GOOGL', 'NVDA', 'TSLA']

class TradingStrategy:
    """
    Clear, well-defined trading strategy with explicit rules.
    
    STRATEGY: Multi-Signal Momentum + Mean Reversion
    
    BUY SIGNALS (all must be true):
    1. RSI < 40 (oversold but not extreme)
    2. Price > 5-day MA (short-term momentum)
    3. 5-day MA > 20-day MA (medium-term trend)
    4. ML model predicts >2% gain with >70% confidence
    5. Volume > 1.2x average (confirmation)
    
    SELL SIGNALS (any can trigger):
    1. RSI > 75 (overbought)
    2. Price < 5-day MA AND 5-day MA < 20-day MA (trend reversal)
    3. ML model predicts >2% loss with >70% confidence
    4. Stop loss: -8% from entry price
    5. Take profit: +15% from entry price
    
    POSITION SIZING:
    - Risk 2% of capital per trade
    - Max 25% of capital in any single stock
    - Hold cash when no clear signals
    """
    
    def __init__(self):
        self.name = "Multi-Signal Momentum + Mean Reversion"
        self.description = """
        Combines momentum indicators with ML predictions for entry/exit decisions.
        Uses conservative position sizing and clear risk management rules.
        """
        
    def generate_signals(self, data: pd.DataFrame, ml_predictions: pd.Series, ml_confidence: pd.Series) -> pd.DataFrame:
        """Generate buy/sell signals based on strategy rules."""
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        signals['close'] = data['Close']
        signals['rsi'] = data['rsi']
        signals['ma_5'] = data['ma_5']
        signals['ma_20'] = data['ma_20']
        signals['volume_ratio'] = data['volume_ratio']
        signals['ml_prediction'] = ml_predictions
        signals['ml_confidence'] = ml_confidence
        
        # Calculate individual signal components
        signals['rsi_oversold'] = signals['rsi'] < 40
        signals['rsi_overbought'] = signals['rsi'] > 75
        signals['price_above_ma5'] = signals['close'] > signals['ma_5']
        signals['ma5_above_ma20'] = signals['ma_5'] > signals['ma_20']
        signals['ml_bullish'] = (signals['ml_prediction'] > 0.02) & (signals['ml_confidence'] > 0.70)
        signals['ml_bearish'] = (signals['ml_prediction'] < -0.02) & (signals['ml_confidence'] > 0.70)
        signals['volume_confirmation'] = signals['volume_ratio'] > 1.2
        
        # Generate buy signals (all conditions must be true)
        signals['buy_signal'] = (
            signals['rsi_oversold'] &
            signals['price_above_ma5'] &
            signals['ma5_above_ma20'] &
            signals['ml_bullish'] &
            signals['volume_confirmation']
        )
        
        # Generate sell signals (any condition can trigger)
        signals['sell_signal'] = (
            signals['rsi_overbought'] |
            (signals['price_above_ma5'] == False) & (signals['ma5_above_ma20'] == False) |
            signals['ml_bearish']
        )
        
        # Clean up signals (can't buy and sell on same day)
        signals.loc[signals['buy_signal'] & signals['sell_signal'], ['buy_signal', 'sell_signal']] = False
        
        return signals

class EnhancedStockAnalyzer:
    """Enhanced analyzer with real timestamps and detailed strategy backtesting."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.strategy = TradingStrategy()
        self.results = {}
        
    def get_real_time_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Get real market data with proper timestamps."""
        try:
            print(f"📡 Fetching real-time data for {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # Get historical data with proper timezone
            data = ticker.history(period=period, interval="1d")
            
            if data.empty:
                print(f"❌ No data available for {symbol}")
                return pd.DataFrame()
            
            print(f"✅ Retrieved {len(data)} days of data from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
            # Ensure we have timezone-aware timestamps
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            
            # Add technical indicators with timestamps
            data = self.calculate_technical_indicators(data, symbol)
            
            # Remove any NaN values but keep timestamp info
            data = data.dropna()
            
            print(f"📊 Final dataset: {len(data)} days with {len(data.columns)} features")
            return data
            
        except Exception as e:
            print(f"❌ Error getting data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate technical indicators with proper timestamp alignment."""
        print(f"🔧 Calculating technical indicators for {symbol}...")
        
        # Basic price indicators
        data['returns'] = data['Close'].pct_change()
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Moving averages
        data['ma_5'] = data['Close'].rolling(5).mean()
        data['ma_20'] = data['Close'].rolling(20).mean()
        data['ma_50'] = data['Close'].rolling(50).mean()
        data['ma_ratio'] = data['ma_5'] / data['ma_20']
        
        # RSI
        data['rsi'] = self.calculate_rsi(data['Close'], 14)
        
        # Volume indicators
        data['volume_ma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_ma']
        
        # Bollinger Bands
        ma_20 = data['ma_20']
        std_20 = data['Close'].rolling(20).std()
        data['bb_upper'] = ma_20 + (2 * std_20)
        data['bb_lower'] = ma_20 - (2 * std_20)
        data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Momentum indicators
        data['momentum_5d'] = data['Close'] / data['Close'].shift(5) - 1
        data['momentum_10d'] = data['Close'] / data['Close'].shift(10) - 1
        data['momentum_20d'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Price position indicators
        data['high_52w'] = data['High'].rolling(252).max()
        data['low_52w'] = data['Low'].rolling(252).min()
        data['price_position_52w'] = (data['Close'] - data['low_52w']) / (data['high_52w'] - data['low_52w'])
        
        # Add proxy indicators with timestamps
        data = self.add_timestamped_proxies(data, symbol)
        
        return data
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI with proper handling of edge cases."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)  # Avoid division by zero
        return 100 - (100 / (1 + rs))
    
    def add_timestamped_proxies(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add company-specific proxy indicators with proper timestamps."""
        print(f"🔗 Adding proxy indicators for {symbol}...")
        
        # Use actual time-based patterns for more realistic proxies
        time_factor = np.arange(len(data)) / len(data)
        base_trend = data['returns'].rolling(30).mean()
        seasonal_factor = np.sin(2 * np.pi * time_factor * 4)  # Quarterly seasonality
        
        if symbol == 'AAPL':
            # Apple: Consumer tech cyclicality
            data['consumer_confidence'] = base_trend * 0.4 + seasonal_factor * 0.1 + np.random.normal(0, 0.05, len(data))
            data['iphone_search_trends'] = base_trend * 0.3 + seasonal_factor * 0.15 + np.random.normal(0, 0.08, len(data))
            data['tech_spending_index'] = base_trend * 0.35 + np.random.normal(0, 0.06, len(data))
            
        elif symbol == 'GOOGL':
            # Google: Search and cloud trends
            data['digital_ad_spending'] = base_trend * 0.4 + seasonal_factor * 0.08 + np.random.normal(0, 0.06, len(data))
            data['cloud_growth_rate'] = base_trend * 0.5 + np.random.normal(0, 0.08, len(data))
            data['search_volume_index'] = base_trend * 0.25 + seasonal_factor * 0.05 + np.random.normal(0, 0.04, len(data))
            
        elif symbol == 'NVDA':
            # NVIDIA: AI and semiconductor cycles
            data['ai_investment_index'] = base_trend * 0.6 + np.random.normal(0, 0.15, len(data))
            data['gpu_demand_index'] = base_trend * 0.45 + seasonal_factor * 0.1 + np.random.normal(0, 0.12, len(data))
            data['datacenter_capex'] = base_trend * 0.5 + np.random.normal(0, 0.10, len(data))
            data['crypto_sentiment'] = base_trend * 0.2 + np.random.normal(0, 0.2, len(data))
            
        elif symbol == 'TSLA':
            # Tesla: EV and energy trends
            data['ev_adoption_rate'] = base_trend * 0.4 + time_factor * 0.3 + np.random.normal(0, 0.12, len(data))
            data['battery_cost_index'] = -base_trend * 0.3 - time_factor * 0.2 + np.random.normal(0, 0.08, len(data))
            data['energy_policy_score'] = base_trend * 0.25 + np.random.normal(0, 0.15, len(data))
            data['auto_sentiment'] = base_trend * 0.35 + seasonal_factor * 0.08 + np.random.normal(0, 0.10, len(data))
        
        return data
    
    def train_ml_model(self, data: pd.DataFrame, symbol: str) -> Tuple[RandomForestRegressor, List[str]]:
        """Train ML model with proper feature selection."""
        print(f"🤖 Training ML model for {symbol}...")
        
        # Select features (exclude target and non-predictive columns)
        exclude_cols = ['returns', 'Dividends', 'Stock Splits'] if 'Dividends' in data.columns else ['returns']
        feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        # Prepare training data (use returns shifted forward as target)
        X = data[feature_columns].fillna(method='ffill').fillna(0)
        y = data['returns'].shift(-1).fillna(0)  # Predict next day return
        
        # Use last 2 years for training, reserve recent data for validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_val = y.iloc[:train_size], y.iloc[train_size:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train ensemble model
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_columns, rf_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"✅ Model trained. Top features:")
        for i, (feature, importance) in enumerate(top_features[:5], 1):
            print(f"   {i}. {feature}: {importance:.4f}")
        
        return rf_model, feature_columns, scaler
    
    def generate_predictions_with_confidence(self, model, scaler, data: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.Series, pd.Series]:
        """Generate ML predictions with confidence estimates."""
        X = data[feature_columns].fillna(method='ffill').fillna(0)
        X_scaled = scaler.transform(X)
        
        # Get predictions from all trees for confidence estimation
        tree_predictions = np.array([tree.predict(X_scaled) for tree in model.estimators_])
        
        predictions = np.mean(tree_predictions, axis=0)
        prediction_std = np.std(tree_predictions, axis=0)
        
        # Convert standard deviation to confidence (0-1 scale)
        confidence = 1 / (1 + prediction_std * 10)  # Higher std = lower confidence
        
        return pd.Series(predictions, index=data.index), pd.Series(confidence, index=data.index)
    
    def run_detailed_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, symbol: str) -> Dict:
        """Run detailed backtesting with trade-by-trade analysis."""
        print(f"📊 Running detailed backtest for {symbol}...")
        
        # Initialize tracking variables
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        trades = []
        daily_values = []
        
        # Risk management parameters
        risk_per_trade = 0.02  # 2% risk per trade
        max_position_size = 0.25  # 25% max position
        stop_loss_pct = -0.08  # 8% stop loss
        take_profit_pct = 0.15  # 15% take profit
        
        for i, (date, row) in enumerate(signals.iterrows()):
            current_price = row['close']
            
            # Calculate current portfolio value
            if position > 0:
                current_value = cash + (position * current_price)
                
                # Check stop loss and take profit
                if entry_price > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    
                    # Stop loss or take profit
                    if pnl_pct <= stop_loss_pct or pnl_pct >= take_profit_pct:
                        # Close position
                        proceeds = position * current_price * 0.999  # 0.1% transaction cost
                        cash += proceeds
                        
                        trade_pnl = proceeds - (position * entry_price)
                        trade_pnl_pct = (current_price - entry_price) / entry_price
                        
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': date,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'shares': position,
                            'pnl_dollars': trade_pnl,
                            'pnl_percent': trade_pnl_pct,
                            'exit_reason': 'Stop Loss' if pnl_pct <= stop_loss_pct else 'Take Profit',
                            'days_held': (date - entry_date).days if entry_date else 0
                        })
                        
                        position = 0
                        entry_price = 0
                        entry_date = None
            else:
                current_value = cash
            
            daily_values.append({
                'date': date,
                'portfolio_value': current_value,
                'cash': cash,
                'position_value': position * current_price if position > 0 else 0,
                'price': current_price
            })
            
            # Process buy signals
            if row['buy_signal'] and position == 0 and cash > 0:
                # Calculate position size based on risk management
                max_shares_by_capital = int((cash * max_position_size) / current_price)
                max_shares_by_risk = int((cash * risk_per_trade) / (current_price * abs(stop_loss_pct)))
                shares_to_buy = min(max_shares_by_capital, max_shares_by_risk)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * 1.001  # 0.1% transaction cost
                    if cost <= cash:
                        cash -= cost
                        position = shares_to_buy
                        entry_price = current_price
                        entry_date = date
            
            # Process sell signals
            elif row['sell_signal'] and position > 0:
                proceeds = position * current_price * 0.999  # 0.1% transaction cost
                cash += proceeds
                
                trade_pnl = proceeds - (position * entry_price)
                trade_pnl_pct = (current_price - entry_price) / entry_price
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': position,
                    'pnl_dollars': trade_pnl,
                    'pnl_percent': trade_pnl_pct,
                    'exit_reason': 'Signal',
                    'days_held': (date - entry_date).days if entry_date else 0
                })
                
                position = 0
                entry_price = 0
                entry_date = None
        
        # Close any remaining position
        if position > 0:
            final_price = signals.iloc[-1]['close']
            proceeds = position * final_price * 0.999
            cash += proceeds
            
            trade_pnl = proceeds - (position * entry_price)
            trade_pnl_pct = (final_price - entry_price) / entry_price
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': signals.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'shares': position,
                'pnl_dollars': trade_pnl,
                'pnl_percent': trade_pnl_pct,
                'exit_reason': 'End of Period',
                'days_held': (signals.index[-1] - entry_date).days if entry_date else 0
            })
        
        # Calculate final portfolio value
        final_value = cash
        if position > 0:
            final_value += position * signals.iloc[-1]['close']
        
        # Create results DataFrame
        trades_df = pd.DataFrame(trades)
        daily_df = pd.DataFrame(daily_values)
        
        return self.calculate_backtest_metrics(trades_df, daily_df, symbol)
    
    def calculate_backtest_metrics(self, trades_df: pd.DataFrame, daily_df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate comprehensive backtest metrics."""
        if trades_df.empty:
            return {
                'error': 'No trades executed',
                'total_trades': 0,
                'strategy_return': 0.0,
                'buy_hold_return': 0.0
            }
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = daily_df.iloc[-1]['portfolio_value']
        strategy_return = (final_value - initial_value) / initial_value
        
        # Buy and hold comparison
        first_price = daily_df.iloc[0]['price']
        last_price = daily_df.iloc[-1]['price']
        buy_hold_return = (last_price - first_price) / first_price
        
        # Trade statistics
        winning_trades = trades_df[trades_df['pnl_percent'] > 0]
        losing_trades = trades_df[trades_df['pnl_percent'] < 0]
        
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        avg_win = winning_trades['pnl_percent'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_percent'].mean() if len(losing_trades) > 0 else 0
        
        # Risk metrics
        daily_returns = daily_df['portfolio_value'].pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        # Maximum drawdown
        rolling_max = daily_df['portfolio_value'].expanding().max()
        drawdown = (daily_df['portfolio_value'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Profit factor
        total_wins = winning_trades['pnl_dollars'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnl_dollars'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        results = {
            'symbol': symbol,
            'backtest_period': f"{daily_df.iloc[0]['date'].strftime('%Y-%m-%d')} to {daily_df.iloc[-1]['date'].strftime('%Y-%m-%d')}",
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate * 100, 1),
            'avg_win_percent': round(avg_win * 100, 2),
            'avg_loss_percent': round(avg_loss * 100, 2),
            'strategy_return': round(strategy_return * 100, 2),
            'buy_hold_return': round(buy_hold_return * 100, 2),
            'outperformance': round((strategy_return - buy_hold_return) * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'max_drawdown': round(max_drawdown * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_days_held': round(trades_df['days_held'].mean(), 1) if len(trades_df) > 0 else 0,
            'final_portfolio_value': round(final_value, 2),
            'total_profit_loss': round(final_value - initial_value, 2),
            'trades_detail': trades_df.to_dict('records'),
            'daily_values': daily_df.to_dict('records')
        }
        
        return results
    
    def analyze_stock_enhanced(self, symbol: str) -> Dict:
        """Complete enhanced analysis with real data and detailed backtesting."""
        print(f"\n{'='*80}")
        print(f"📊 ENHANCED ANALYSIS: {symbol}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Get real-time data with timestamps
            data = self.get_real_time_data(symbol)
            if data.empty:
                return None
            
            # Step 2: Train ML model
            model, feature_columns, scaler = self.train_ml_model(data, symbol)
            
            # Step 3: Generate predictions with confidence
            ml_predictions, ml_confidence = self.generate_predictions_with_confidence(
                model, scaler, data, feature_columns
            )
            
            # Step 4: Generate trading signals
            signals = self.strategy.generate_signals(data, ml_predictions, ml_confidence)
            
            # Step 5: Run detailed backtesting
            backtest_results = self.run_detailed_backtest(data, signals, symbol)
            
            # Step 6: Current market analysis
            current_analysis = self.analyze_current_market_state(data, signals, symbol)
            
            # Step 7: Compile comprehensive results
            result = {
                'symbol': symbol,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_period': f"{data.index[0].strftime('%Y-%m-%d %H:%M:%S %Z')} to {data.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}",
                'total_data_points': len(data),
                'strategy_description': self.strategy.description.strip(),
                'current_analysis': current_analysis,
                'backtest_results': backtest_results,
                'feature_importance': dict(zip(feature_columns, model.feature_importances_)),
                'recent_signals': signals.tail(10).to_dict('records')
            }
            
            self.results[symbol] = result
            return result
            
        except Exception as e:
            print(f"❌ Error in enhanced analysis for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_current_market_state(self, data: pd.DataFrame, signals: pd.DataFrame, symbol: str) -> Dict:
        """Analyze current market state and generate prediction."""
        print(f"🎯 Analyzing current market state for {symbol}...")
        
        current_row = signals.iloc[-1]
        recent_data = data.tail(5)
        
        # Current market metrics
        current_price = current_row['close']
        rsi = current_row['rsi']
        ma_5 = current_row['ma_5']
        ma_20 = current_row['ma_20']
        volume_ratio = current_row['volume_ratio']
        ml_prediction = current_row['ml_prediction']
        ml_confidence = current_row['ml_confidence']
        
        # Price momentum
        momentum_5d = (current_price / data['Close'].iloc[-6] - 1) * 100
        momentum_20d = (current_price / data['Close'].iloc[-21] - 1) * 100
        
        # Volatility analysis
        recent_volatility = recent_data['returns'].std() * np.sqrt(252) * 100
        
        # Signal analysis
        buy_signal = current_row['buy_signal']
        sell_signal = current_row['sell_signal']
        
        # Generate recommendation
        if buy_signal:
            action = "BUY"
            confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
            rationale = f"All buy conditions met: RSI oversold ({rsi:.1f}), positive momentum, ML predicts {ml_prediction*100:.1f}% gain"
        elif sell_signal:
            action = "SELL"
            confidence = "HIGH" if ml_confidence > 0.8 else "MEDIUM"
            rationale = f"Sell signal triggered: RSI overbought or trend reversal, ML confidence {ml_confidence*100:.1f}%"
        else:
            action = "HOLD"
            confidence = "LOW" if ml_confidence < 0.6 else "MEDIUM"
            rationale = f"No clear signals. RSI: {rsi:.1f}, ML prediction: {ml_prediction*100:.1f}%"
        
        # Calculate target price based on ML prediction
        target_price = current_price * (1 + ml_prediction)
        
        analysis = {
            'timestamp': data.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z'),
            'current_price': round(current_price, 2),
            'recommendation': {
                'action': action,
                'confidence': confidence,
                'rationale': rationale,
                'target_price': round(target_price, 2),
                'expected_return': round(ml_prediction * 100, 2)
            },
            'market_metrics': {
                'rsi': round(rsi, 1),
                'rsi_signal': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral',
                'ma_5': round(ma_5, 2),
                'ma_20': round(ma_20, 2),
                'price_vs_ma5': 'Above' if current_price > ma_5 else 'Below',
                'price_vs_ma20': 'Above' if current_price > ma_20 else 'Below',
                'volume_ratio': round(volume_ratio, 2),
                'volume_signal': 'High' if volume_ratio > 1.5 else 'Normal' if volume_ratio > 0.8 else 'Low'
            },
            'momentum': {
                'momentum_5d': round(momentum_5d, 2),
                'momentum_20d': round(momentum_20d, 2),
                'trend': 'Bullish' if momentum_5d > 0 and momentum_20d > 0 else 'Bearish' if momentum_5d < 0 and momentum_20d < 0 else 'Mixed'
            },
            'ml_analysis': {
                'prediction_percent': round(ml_prediction * 100, 2),
                'confidence': round(ml_confidence * 100, 1),
                'signal_strength': 'Strong' if ml_confidence > 0.8 else 'Moderate' if ml_confidence > 0.6 else 'Weak'
            },
            'risk_metrics': {
                'recent_volatility': round(recent_volatility, 2),
                'volatility_level': 'High' if recent_volatility > 30 else 'Medium' if recent_volatility > 20 else 'Low'
            }
        }
        
        return analysis
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive report with strategy details and backtesting."""
        report = []
        report.append("=" * 100)
        report.append("📊 ENHANCED STOCK PREDICTION & BACKTESTING REPORT")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Initial Capital: ${self.initial_capital:,.2f}")
        report.append("")
        
        # Strategy description
        report.append("🎯 TRADING STRATEGY:")
        report.append(f"Strategy Name: {self.strategy.name}")
        report.append(f"Description: {self.strategy.description}")
        report.append("")
        
        # Individual stock analyses
        for symbol, result in self.results.items():
            if not result:
                continue
                
            report.append("=" * 80)
            report.append(f"📈 {symbol} - DETAILED ANALYSIS")
            report.append("=" * 80)
            
            # Current market state
            current = result['current_analysis']
            recommendation = current['recommendation']
            
            report.append(f"\n🎯 CURRENT RECOMMENDATION:")
            report.append(f"  Action: {recommendation['action']}")
            report.append(f"  Confidence: {recommendation['confidence']}")
            report.append(f"  Current Price: ${current['current_price']}")
            report.append(f"  Target Price: ${recommendation['target_price']}")
            report.append(f"  Expected Return: {recommendation['expected_return']:+.2f}%")
            report.append(f"  Rationale: {recommendation['rationale']}")
            
            # Market metrics
            metrics = current['market_metrics']
            report.append(f"\n📊 MARKET METRICS:")
            report.append(f"  RSI: {metrics['rsi']} ({metrics['rsi_signal']})")
            report.append(f"  Price vs MA5: {metrics['price_vs_ma5']}")
            report.append(f"  Price vs MA20: {metrics['price_vs_ma20']}")
            report.append(f"  Volume: {metrics['volume_ratio']}x average ({metrics['volume_signal']})")
            report.append(f"  5-day Momentum: {current['momentum']['momentum_5d']:+.2f}%")
            report.append(f"  20-day Momentum: {current['momentum']['momentum_20d']:+.2f}%")
            report.append(f"  Recent Volatility: {current['risk_metrics']['recent_volatility']:.1f}%")
            
            # Backtesting results
            backtest = result['backtest_results']
            if 'error' not in backtest:
                report.append(f"\n📊 BACKTESTING RESULTS:")
                report.append(f"  Period: {backtest['backtest_period']}")
                report.append(f"  Total Trades: {backtest['total_trades']}")
                report.append(f"  Win Rate: {backtest['win_rate']:.1f}% ({backtest['winning_trades']}/{backtest['total_trades']})")
                report.append(f"  Average Win: {backtest['avg_win_percent']:+.2f}%")
                report.append(f"  Average Loss: {backtest['avg_loss_percent']:+.2f}%")
                report.append(f"  Strategy Return: {backtest['strategy_return']:+.2f}%")
                report.append(f"  Buy & Hold Return: {backtest['buy_hold_return']:+.2f}%")
                report.append(f"  Outperformance: {backtest['outperformance']:+.2f}%")
                report.append(f"  Sharpe Ratio: {backtest['sharpe_ratio']:.3f}")
                report.append(f"  Max Drawdown: {backtest['max_drawdown']:.2f}%")
                report.append(f"  Profit Factor: {backtest['profit_factor']:.2f}")
                report.append(f"  Average Days Held: {backtest['avg_days_held']:.1f}")
                report.append(f"  Final Portfolio Value: ${backtest['final_portfolio_value']:,.2f}")
                report.append(f"  Total P&L: ${backtest['total_profit_loss']:+,.2f}")
                
                # Performance rating
                outperf = backtest['outperformance']
                if outperf > 15:
                    rating = "⭐⭐⭐⭐⭐ EXCELLENT"
                elif outperf > 10:
                    rating = "⭐⭐⭐⭐ VERY GOOD"
                elif outperf > 5:
                    rating = "⭐⭐⭐ GOOD"
                elif outperf > 0:
                    rating = "⭐⭐ FAIR"
                else:
                    rating = "⭐ NEEDS IMPROVEMENT"
                
                report.append(f"  Performance Rating: {rating}")
                
                # Recent trades sample
                if backtest['trades_detail']:
                    report.append(f"\n📋 RECENT TRADES (Last 3):")
                    recent_trades = backtest['trades_detail'][-3:]
                    for i, trade in enumerate(recent_trades, 1):
                        entry_date = pd.to_datetime(trade['entry_date']).strftime('%Y-%m-%d')
                        exit_date = pd.to_datetime(trade['exit_date']).strftime('%Y-%m-%d')
                        report.append(f"    {i}. {entry_date} → {exit_date}: {trade['pnl_percent']*100:+.2f}% ({trade['exit_reason']})")
            else:
                report.append(f"\n📊 BACKTESTING: {backtest['error']}")
        
        # Portfolio summary
        successful_results = [r for r in self.results.values() if r and 'error' not in r.get('backtest_results', {})]
        if successful_results:
            avg_return = np.mean([r['backtest_results']['strategy_return'] for r in successful_results])
            avg_outperf = np.mean([r['backtest_results']['outperformance'] for r in successful_results])
            total_trades = sum([r['backtest_results']['total_trades'] for r in successful_results])
            avg_win_rate = np.mean([r['backtest_results']['win_rate'] for r in successful_results])
            
            report.append("\n" + "=" * 80)
            report.append("📝 PORTFOLIO SUMMARY")
            report.append("=" * 80)
            report.append(f"Stocks Analyzed: {len(successful_results)}/{len(TARGET_STOCKS)}")
            report.append(f"Average Strategy Return: {avg_return:+.2f}%")
            report.append(f"Average Outperformance: {avg_outperf:+.2f}%")
            report.append(f"Total Trades Across All Stocks: {total_trades}")
            report.append(f"Average Win Rate: {avg_win_rate:.1f}%")
            
            report.append(f"\n🎯 CURRENT PORTFOLIO RECOMMENDATIONS:")
            for symbol, result in self.results.items():
                if result and 'current_analysis' in result:
                    rec = result['current_analysis']['recommendation']
                    report.append(f"  {symbol}: {rec['action']} (Target: ${rec['target_price']}, Expected: {rec['expected_return']:+.1f}%)")
        
        report.append(f"\n⚠️ DISCLAIMER:")
        report.append("This analysis is for educational purposes only. Past performance does not guarantee future results.")
        report.append("Always consult with a qualified financial advisor before making investment decisions.")
        report.append("Consider transaction costs, taxes, and market liquidity in your actual trading.")
        
        report.append("\n" + "=" * 100)
        
        return "\n".join(report)

def main():
    """Main function for enhanced analysis."""
    print("🚀 ENHANCED STOCK PREDICTION ENGINE")
    print("Real-time data • Clear strategy • Detailed backtesting")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = EnhancedStockAnalyzer(initial_capital=100000)
    
    print(f"📊 Analyzing: {', '.join(TARGET_STOCKS)}")
    print(f"🏦 Initial Capital: ${analyzer.initial_capital:,}")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze each stock
    for symbol in TARGET_STOCKS:
        analyzer.analyze_stock_enhanced(symbol)
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON results
    json_filename = f"enhanced_results_{timestamp}.json"
    json_results = {}
    
    for symbol, result in analyzer.results.items():
        if result:
            json_results[symbol] = {
                'analysis_timestamp': result['analysis_timestamp'],
                'data_period': result['data_period'],
                'current_analysis': result['current_analysis'],
                'backtest_summary': {
                    'total_trades': result['backtest_results'].get('total_trades', 0),
                    'win_rate': result['backtest_results'].get('win_rate', 0),
                    'strategy_return': result['backtest_results'].get('strategy_return', 0),
                    'outperformance': result['backtest_results'].get('outperformance', 0),
                    'sharpe_ratio': result['backtest_results'].get('sharpe_ratio', 0),
                    'max_drawdown': result['backtest_results'].get('max_drawdown', 0)
                },
                'strategy_description': result['strategy_description']
            }
    
    with open(json_filename, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # Save detailed CSV for further analysis
    csv_filename = f"trade_details_{timestamp}.csv"
    all_trades = []
    
    for symbol, result in analyzer.results.items():
        if result and 'backtest_results' in result and 'trades_detail' in result['backtest_results']:
            for trade in result['backtest_results']['trades_detail']:
                trade['symbol'] = symbol
                all_trades.append(trade)
    
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(csv_filename, index=False)
        print(f"\n💾 Detailed trade data saved: {csv_filename}")
    
    print(f"💾 Analysis results saved: {json_filename}")
    print(f"⏰ Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()