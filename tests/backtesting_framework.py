# tests/backtesting_framework.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from features.data_ingestion.pipeline import build_dataset
from features.modeling.predictor import EnsemblePredictor
from features.trading_decision.guidance import TradingGuidanceEngine, AdvancedGuidanceEngine
from features.rl_training.environment import TradingEnvironment

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    volatility: float
    calmar_ratio: float
    trades: List[Dict]
    equity_curve: pd.Series
    metrics_summary: Dict

class BacktestingFramework:
    """
    Comprehensive backtesting framework for the prediction engine.
    Supports walk-forward analysis, Monte Carlo simulation, and noise resilience testing.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Backtest state
        self.reset_state()
        
        # Results storage
        self.backtest_results = {}
    
    def reset_state(self):
        """Reset backtesting state."""
        self.current_capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self.current_trade = None
        
    def run_backtest(self, 
                    stock: str,
                    start_date: str = '2022-01-01',
                    end_date: str = '2024-01-01',
                    prediction_window: int = 30,
                    rebalance_freq: str = 'daily',
                    noise_level: float = 0.0) -> BacktestResult:
        """
        Run a comprehensive backtest for a given stock.
        
        Args:
            stock: Stock symbol to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            prediction_window: Number of days to use for prediction
            rebalance_freq: Frequency of rebalancing ('daily', 'weekly')
            noise_level: Level of noise to inject for robustness testing
        """
        print(f"Running backtest for {stock}: {start_date} to {end_date}")
        
        # Reset state
        self.reset_state()
        
        # Build dataset
        data = build_dataset(stock, start_date=start_date)
        if data.empty:
            raise ValueError(f"No data available for {stock}")
        
        # Filter data by end_date
        data = data[data.index <= end_date]
        
        # Initialize components
        guidance_engine = TradingGuidanceEngine()
        
        # Walk-forward analysis
        results = self._walk_forward_backtest(
            data, guidance_engine, prediction_window, rebalance_freq, noise_level
        )
        
        # Calculate performance metrics
        backtest_result = self._calculate_metrics(results)
        
        # Store results
        self.backtest_results[f"{stock}_{start_date}_{end_date}"] = backtest_result
        
        return backtest_result
    
    def _walk_forward_backtest(self, 
                              data: pd.DataFrame,
                              guidance_engine: TradingGuidanceEngine,
                              prediction_window: int,
                              rebalance_freq: str,
                              noise_level: float) -> Dict:
        """Perform walk-forward backtesting."""
        
        results = {
            'trades': [],
            'equity_curve': [],
            'predictions': [],
            'daily_returns': []
        }
        
        # Determine rebalance frequency
        rebalance_days = {'daily': 1, 'weekly': 5, 'monthly': 22}
        freq_days = rebalance_days.get(rebalance_freq, 1)
        
        # Start backtesting from prediction_window + 10 to have enough training data
        start_idx = prediction_window + 10
        
        for i in range(start_idx, len(data), freq_days):
            try:
                # Get training data (up to current point)
                train_data = data.iloc[:i]
                
                # Get current price for execution
                current_price = data['Close'].iloc[i]
                
                # Add noise if specified
                if noise_level > 0:
                    price_noise = np.random.normal(0, noise_level * current_price)
                    execution_price = max(0.01, current_price + price_noise)
                else:
                    execution_price = current_price
                
                # Skip if insufficient training data
                if len(train_data) < prediction_window:
                    continue
                
                # Create predictor and make prediction
                try:
                    predictor = EnsemblePredictor(train_data, target_col='returns')
                    
                    # Use last prediction_window days for prediction
                    pred_data = train_data.iloc[-prediction_window:]
                    pred_mean, pred_std = predictor.predict(pred_data)
                    
                    confidence = 1 / (1 + pred_std) if pred_std > 0 else 0.5
                    volatility = train_data['volatility'].iloc[-1] if 'volatility' in train_data else 0.02
                    
                    # Get trading guidance
                    action, rationale, metrics = guidance_engine.get_guidance(
                        pred_mean, confidence, volatility
                    )
                    
                    # Record prediction
                    results['predictions'].append({
                        'date': data.index[i],
                        'predicted_change': pred_mean,
                        'confidence': confidence,
                        'action': action,
                        'price': execution_price
                    })
                    
                    # Execute trade
                    trade_result = self._execute_trade(action, execution_price, data.index[i])
                    if trade_result:
                        results['trades'].append(trade_result)
                    
                except Exception as e:
                    print(f"Prediction failed at index {i}: {str(e)}")
                    # Default to hold action
                    action = "Hold"
                
                # Record equity
                portfolio_value = self.current_capital + self.position * execution_price
                results['equity_curve'].append({
                    'date': data.index[i],
                    'portfolio_value': portfolio_value,
                    'cash': self.current_capital,
                    'position_value': self.position * execution_price,
                    'price': execution_price
                })
                
                # Calculate daily return
                if len(results['equity_curve']) > 1:
                    prev_value = results['equity_curve'][-2]['portfolio_value']
                    daily_return = (portfolio_value - prev_value) / prev_value
                    results['daily_returns'].append(daily_return)
                
            except Exception as e:
                print(f"Error in backtest at index {i}: {str(e)}")
                continue
        
        return results
    
    def _execute_trade(self, action: str, price: float, date) -> Optional[Dict]:
        """Execute a trading action."""
        
        # Apply slippage
        if action in ["Buy", "Sell"]:
            slippage_adjustment = self.slippage * price
            if action == "Buy":
                execution_price = price + slippage_adjustment
            else:
                execution_price = price - slippage_adjustment
        else:
            return None  # No trade for Hold
        
        trade_result = None
        
        if action == "Buy" and self.position == 0:
            # Buy signal - enter long position
            max_shares = int(self.current_capital * 0.95 / (execution_price * (1 + self.transaction_cost)))
            
            if max_shares > 0:
                total_cost = max_shares * execution_price * (1 + self.transaction_cost)
                
                if total_cost <= self.current_capital:
                    self.current_capital -= total_cost
                    self.position = max_shares
                    
                    trade_result = {
                        'date': date,
                        'action': 'buy',
                        'shares': max_shares,
                        'price': execution_price,
                        'cost': total_cost,
                        'portfolio_value': self.current_capital + self.position * execution_price
                    }
                    
                    self.current_trade = {
                        'entry_date': date,
                        'entry_price': execution_price,
                        'shares': max_shares
                    }
        
        elif action == "Sell" and self.position > 0:
            # Sell signal - exit long position
            proceeds = self.position * execution_price * (1 - self.transaction_cost)
            
            trade_result = {
                'date': date,
                'action': 'sell',
                'shares': self.position,
                'price': execution_price,
                'proceeds': proceeds,
                'portfolio_value': self.current_capital + proceeds
            }
            
            # Calculate trade metrics if we have entry info
            if self.current_trade:
                entry_price = self.current_trade['entry_price']
                trade_return = (execution_price - entry_price) / entry_price
                hold_days = (date - self.current_trade['entry_date']).days
                
                trade_result.update({
                    'entry_date': self.current_trade['entry_date'],
                    'entry_price': entry_price,
                    'trade_return': trade_return,
                    'hold_days': hold_days,
                    'profit_loss': proceeds - (self.position * entry_price)
                })
            
            self.current_capital += proceeds
            self.position = 0
            self.current_trade = None
        
        return trade_result
    
    def _calculate_metrics(self, results: Dict) -> BacktestResult:
        """Calculate comprehensive performance metrics."""
        
        if not results['equity_curve']:
            return BacktestResult(
                total_return=0, annualized_return=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0,
                total_trades=0, avg_trade_duration=0, volatility=0,
                calmar_ratio=0, trades=[], equity_curve=pd.Series(),
                metrics_summary={}
            )
        
        # Convert to DataFrames
        equity_df = pd.DataFrame(results['equity_curve'])
        trades_df = pd.DataFrame(results['trades']) if results['trades'] else pd.DataFrame()
        
        # Basic return metrics
        initial_value = equity_df['portfolio_value'].iloc[0]
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Annualized return
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Daily returns for risk metrics
        daily_returns = np.array(results['daily_returns']) if results['daily_returns'] else np.array([0])
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = daily_returns - (risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Maximum drawdown
        equity_series = equity_df['portfolio_value']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Trade-based metrics
        if not trades_df.empty:
            completed_trades = trades_df[trades_df['action'] == 'sell']
            
            if not completed_trades.empty:
                win_rate = (completed_trades['trade_return'] > 0).mean()
                
                winning_trades = completed_trades[completed_trades['trade_return'] > 0]['trade_return']
                losing_trades = completed_trades[completed_trades['trade_return'] < 0]['trade_return']
                
                avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
                avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 1
                
                profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
                avg_trade_duration = completed_trades['hold_days'].mean()
                total_trades = len(completed_trades)
            else:
                win_rate = 0
                profit_factor = 0
                avg_trade_duration = 0
                total_trades = 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_duration = 0
            total_trades = 0
        
        # Volatility
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Create equity curve series
        equity_curve = pd.Series(
            equity_df['portfolio_value'].values,
            index=pd.to_datetime(equity_df['date'])
        )
        
        # Metrics summary
        metrics_summary = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_days': days,
            'trading_days': len(equity_df),
            'transaction_cost': self.transaction_cost,
            'slippage': self.slippage
        }
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=avg_trade_duration,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            trades=results['trades'],
            equity_curve=equity_curve,
            metrics_summary=metrics_summary
        )
    
    def run_monte_carlo_backtest(self, 
                                stock: str,
                                n_simulations: int = 100,
                                noise_levels: List[float] = [0.0, 0.01, 0.02, 0.03],
                                **backtest_params) -> Dict:
        """
        Run Monte Carlo simulations with different noise levels to test robustness.
        """
        print(f"Running Monte Carlo backtest with {n_simulations} simulations...")
        
        results = {
            'simulations': [],
            'noise_analysis': {},
            'summary_stats': {}
        }
        
        for noise_level in noise_levels:
            noise_results = []
            
            print(f"Testing noise level: {noise_level:.1%}")
            
            for sim in range(n_simulations):
                try:
                    # Set random seed for reproducibility within noise level
                    np.random.seed(sim + int(noise_level * 1000))
                    
                    # Run backtest with noise
                    backtest_result = self.run_backtest(
                        stock=stock,
                        noise_level=noise_level,
                        **backtest_params
                    )
                    
                    noise_results.append({
                        'simulation': sim,
                        'noise_level': noise_level,
                        'total_return': backtest_result.total_return,
                        'sharpe_ratio': backtest_result.sharpe_ratio,
                        'max_drawdown': backtest_result.max_drawdown,
                        'win_rate': backtest_result.win_rate,
                        'total_trades': backtest_result.total_trades
                    })
                    
                except Exception as e:
                    print(f"Simulation {sim} failed for noise {noise_level}: {str(e)}")
                    continue
            
            # Analyze results for this noise level
            if noise_results:
                noise_df = pd.DataFrame(noise_results)
                results['noise_analysis'][noise_level] = {
                    'mean_return': noise_df['total_return'].mean(),
                    'std_return': noise_df['total_return'].std(),
                    'mean_sharpe': noise_df['sharpe_ratio'].mean(),
                    'mean_drawdown': noise_df['max_drawdown'].mean(),
                    'success_rate': (noise_df['total_return'] > 0).mean(),
                    'results': noise_results
                }
            
            results['simulations'].extend(noise_results)
        
        # Overall summary statistics
        if results['simulations']:
            all_results_df = pd.DataFrame(results['simulations'])
            results['summary_stats'] = {
                'total_simulations': len(all_results_df),
                'overall_mean_return': all_results_df['total_return'].mean(),
                'overall_std_return': all_results_df['total_return'].std(),
                'noise_resilience_score': self._calculate_noise_resilience(results['noise_analysis'])
            }
        
        return results
    
    def _calculate_noise_resilience(self, noise_analysis: Dict) -> float:
        """Calculate a noise resilience score."""
        if len(noise_analysis) < 2:
            return 0.0
        
        # Compare performance degradation as noise increases
        noise_levels = sorted(noise_analysis.keys())
        base_return = noise_analysis[noise_levels[0]]['mean_return']
        
        degradation_scores = []
        for noise_level in noise_levels[1:]:
            noisy_return = noise_analysis[noise_level]['mean_return']
            degradation = abs(noisy_return - base_return) / (abs(base_return) + 0.01)
            degradation_scores.append(1 - min(1.0, degradation))
        
        return np.mean(degradation_scores)
    
    def compare_strategies(self, 
                          stock: str,
                          strategies: List[str] = ['basic', 'advanced'],
                          **backtest_params) -> Dict:
        """Compare different trading strategies."""
        
        comparison_results = {}
        
        for strategy in strategies:
            print(f"Testing strategy: {strategy}")
            
            # Modify guidance engine based on strategy
            if strategy == 'basic':
                # Use basic guidance engine (already default)
                pass
            elif strategy == 'advanced':
                # Would use AdvancedGuidanceEngine with ensemble voting
                pass
            
            try:
                result = self.run_backtest(stock=stock, **backtest_params)
                comparison_results[strategy] = result
            except Exception as e:
                print(f"Strategy {strategy} failed: {str(e)}")
                comparison_results[strategy] = None
        
        return comparison_results
    
    def plot_backtest_results(self, result: BacktestResult, save_path: str = None):
        """Plot comprehensive backtest results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        axes[0, 0].plot(result.equity_curve.index, result.equity_curve.values)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Drawdown
        running_max = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - running_max) / running_max
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.7, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].grid(True)
        
        # Performance metrics bar chart
        metrics = {
            'Total Return': f"{result.total_return:.1%}",
            'Sharpe Ratio': f"{result.sharpe_ratio:.2f}",
            'Max Drawdown': f"{result.max_drawdown:.1%}",
            'Win Rate': f"{result.win_rate:.1%}"
        }
        
        axes[1, 0].bar(range(len(metrics)), [
            result.total_return * 100,
            result.sharpe_ratio * 10,  # Scale for visibility
            result.max_drawdown * 100,
            result.win_rate * 100
        ])
        axes[1, 0].set_xticks(range(len(metrics)))
        axes[1, 0].set_xticklabels(metrics.keys(), rotation=45)
        axes[1, 0].set_title('Performance Metrics')
        axes[1, 0].set_ylabel('Value')
        
        # Trade analysis
        if result.trades:
            trades_df = pd.DataFrame(result.trades)
            completed_trades = trades_df[trades_df['action'] == 'sell']
            
            if not completed_trades.empty:
                axes[1, 1].hist(completed_trades['trade_return'] * 100, bins=20, alpha=0.7)
                axes[1, 1].set_title('Trade Return Distribution')
                axes[1, 1].set_xlabel('Return (%)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].axvline(x=0, color='red', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Backtest plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, result: BacktestResult, stock: str) -> str:
        """Generate a comprehensive backtest report."""
        
        report = f"""
BACKTEST REPORT - {stock}
{'='*50}

PERFORMANCE SUMMARY
Total Return: {result.total_return:.2%}
Annualized Return: {result.annualized_return:.2%}
Sharpe Ratio: {result.sharpe_ratio:.3f}
Calmar Ratio: {result.calmar_ratio:.3f}

RISK METRICS
Maximum Drawdown: {result.max_drawdown:.2%}
Volatility: {result.volatility:.2%}

TRADING METRICS
Total Trades: {result.total_trades}
Win Rate: {result.win_rate:.2%}
Profit Factor: {result.profit_factor:.3f}
Average Trade Duration: {result.avg_trade_duration:.1f} days

CONFIGURATION
Initial Capital: ${result.metrics_summary.get('initial_capital', 0):,.2f}
Final Value: ${result.metrics_summary.get('final_value', 0):,.2f}
Transaction Cost: {result.metrics_summary.get('transaction_cost', 0):.1%}
Slippage: {result.metrics_summary.get('slippage', 0):.1%}

ANALYSIS
"""
        
        # Add performance analysis
        if result.sharpe_ratio > 1.0:
            report += "✓ Excellent risk-adjusted returns (Sharpe > 1.0)\n"
        elif result.sharpe_ratio > 0.5:
            report += "○ Good risk-adjusted returns (Sharpe > 0.5)\n"
        else:
            report += "✗ Poor risk-adjusted returns (Sharpe < 0.5)\n"
        
        if result.max_drawdown < 0.1:
            report += "✓ Low maximum drawdown (<10%)\n"
        elif result.max_drawdown < 0.2:
            report += "○ Moderate maximum drawdown (<20%)\n"
        else:
            report += "✗ High maximum drawdown (>20%)\n"
        
        if result.win_rate > 0.6:
            report += "✓ High win rate (>60%)\n"
        elif result.win_rate > 0.4:
            report += "○ Moderate win rate (>40%)\n"
        else:
            report += "✗ Low win rate (<40%)\n"
        
        return report

# Usage example
if __name__ == '__main__':
    # Initialize backtesting framework
    backtester = BacktestingFramework(initial_capital=10000)
    
    # Run a simple backtest
    print("Running backtest for AAPL...")
    try:
        result = backtester.run_backtest(
            stock='AAPL',
            start_date='2023-01-01',
            end_date='2024-01-01',
            prediction_window=30,
            rebalance_freq='weekly'
        )
        
        # Print results
        print("\nBacktest Results:")
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {result.max_drawdown:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        
        # Generate full report
        report = backtester.generate_report(result, 'AAPL')
        print(report)
        
        # Plot results
        backtester.plot_backtest_results(result)
        
    except Exception as e:
        print(f"Backtest failed: {str(e)}")
    
    # Run Monte Carlo simulation (smaller scale for demo)
    print("\nRunning Monte Carlo simulation...")
    try:
        mc_results = backtester.run_monte_carlo_backtest(
            stock='AAPL',
            n_simulations=10,  # Small number for demo
            noise_levels=[0.0, 0.01, 0.02],
            start_date='2023-06-01',
            end_date='2023-12-01'
        )
        
        print("Monte Carlo Results:")
        for noise_level, analysis in mc_results['noise_analysis'].items():
            print(f"Noise {noise_level:.1%}: Return {analysis['mean_return']:.2%} ± {analysis['std_return']:.2%}")
        
        resilience_score = mc_results['summary_stats']['noise_resilience_score']
        print(f"Noise Resilience Score: {resilience_score:.3f}")
        
    except Exception as e:
        print(f"Monte Carlo simulation failed: {str(e)}")