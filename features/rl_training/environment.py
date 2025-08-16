# features/rl_training/environment.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

class TradingEnvironment(gym.Env):
    """
    Enhanced trading environment for RL-based strategy optimization.
    Supports noise injection, market regime simulation, and multi-asset trading.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.001,
                 noise_level: float = 0.02,
                 regime_aware: bool = True):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.noise_level = noise_level
        self.regime_aware = regime_aware
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [price_features, volatility, proxy_features, portfolio_state]
        obs_dim = len(data.columns) + 3  # +3 for balance, position, portfolio_value
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        
        # Environment state
        self.reset()
        
        # Performance tracking
        self.trade_history = []
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'noise_resilience_score': 0.0
        }
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # Number of shares held
        self.portfolio_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        
        # Market regime detection
        if self.regime_aware:
            self.current_regime = self._detect_market_regime(0)
        else:
            self.current_regime = 'neutral'
        
        # Noise injection for robustness training
        self.noise_factor = np.random.normal(1.0, self.noise_level)
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment."""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        # Get current and next prices
        current_price = self._get_noisy_price(self.current_step)
        next_price = self._get_noisy_price(self.current_step + 1)
        
        # Execute action
        reward = self._execute_action(action, current_price, next_price)
        
        # Update state
        self.current_step += 1
        self.portfolio_value = self.balance + self.position * next_price
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        
        # Update market regime
        if self.regime_aware:
            self.current_regime = self._detect_market_regime(self.current_step)
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1) or (self.portfolio_value <= 0.1 * self.initial_balance)
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'regime': self.current_regime,
            'price': next_price
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation state."""
        if self.current_step >= len(self.data):
            # Return zeros if we're past the data
            return np.zeros(self.observation_space.shape[0])
        
        # Market data features
        market_features = self.data.iloc[self.current_step].values
        
        # Portfolio state
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position,  # Current position
            self.portfolio_value / self.initial_balance  # Normalized portfolio value
        ])
        
        # Combine all features
        observation = np.concatenate([market_features, portfolio_features])
        
        # Handle NaN values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return observation.astype(np.float32)
    
    def _get_noisy_price(self, step):
        """Get price with noise injection for robustness."""
        if step >= len(self.data):
            return self.data['Close'].iloc[-1]
        
        base_price = self.data['Close'].iloc[step]
        
        # Add market microstructure noise
        noise = np.random.normal(0, self.noise_level * base_price)
        
        # Add regime-dependent noise
        regime_noise = self._get_regime_noise(base_price)
        
        return max(0.01, base_price + noise + regime_noise)
    
    def _get_regime_noise(self, price):
        """Add regime-specific noise patterns."""
        if self.current_regime == 'bull':
            # Bull markets: less downside noise, occasional sharp moves up
            return np.random.normal(0.001 * price, 0.005 * price)
        elif self.current_regime == 'bear':
            # Bear markets: more downside pressure, higher volatility
            return np.random.normal(-0.002 * price, 0.01 * price)
        else:
            # Neutral: symmetric noise
            return np.random.normal(0, 0.007 * price)
    
    def _detect_market_regime(self, step):
        """Simple market regime detection."""
        if step < 20:
            return 'neutral'
        
        # Look at recent price movements
        recent_data = self.data['Close'].iloc[max(0, step-19):step+1]
        
        if len(recent_data) < 2:
            return 'neutral'
        
        price_change = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1
        volatility = recent_data.pct_change().std()
        
        if price_change > 0.1 and volatility < 0.03:
            return 'bull'
        elif price_change < -0.1 or volatility > 0.05:
            return 'bear'
        else:
            return 'neutral'
    
    def _execute_action(self, action, current_price, next_price):
        """Execute trading action and calculate reward."""
        # Record trade
        trade_record = {
            'step': self.current_step,
            'action': action,
            'price': current_price,
            'portfolio_value_before': self.portfolio_value,
            'regime': self.current_regime
        }
        
        reward = 0
        actual_return = (next_price - current_price) / current_price
        
        if action == 1:  # Buy
            if self.balance > current_price * (1 + self.transaction_cost):
                shares_to_buy = int(self.balance / (current_price * (1 + self.transaction_cost)))
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.position += shares_to_buy
                
                # Reward based on next period return
                if actual_return > 0:
                    reward = actual_return * 10  # Scale reward
                else:
                    reward = actual_return * 5  # Less penalty for wrong direction
                
                # Penalty for high noise periods
                if abs(actual_return) < self.noise_level:
                    reward -= 0.1  # Penalty for trading in noise
                
                trade_record['executed'] = True
                trade_record['shares'] = shares_to_buy
            else:
                reward = -0.01  # Small penalty for invalid action
                trade_record['executed'] = False
        
        elif action == 2:  # Sell
            if self.position > 0:
                proceeds = self.position * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                
                # Reward based on avoiding losses
                if actual_return < 0:
                    reward = -actual_return * 10  # Reward for avoiding loss
                else:
                    reward = -actual_return * 5  # Penalty for missing gain
                
                # Bonus for selling before major drops
                if actual_return < -0.02:
                    reward += 0.5
                
                trade_record['executed'] = True
                trade_record['shares'] = -self.position
                self.position = 0
            else:
                reward = -0.01  # Small penalty for invalid action
                trade_record['executed'] = False
        
        else:  # Hold (action == 0)
            # Small reward for holding during low-signal periods
            if abs(actual_return) < self.noise_level:
                reward = 0.02  # Reward for not trading in noise
            else:
                # Opportunity cost for missing large moves
                reward = -abs(actual_return) * 2
        
        # Additional regime-aware rewards
        if self.regime_aware:
            reward += self._get_regime_reward(action, actual_return)
        
        trade_record['reward'] = reward
        trade_record['actual_return'] = actual_return
        self.trade_history.append(trade_record)
        
        return reward
    
    def _get_regime_reward(self, action, actual_return):
        """Additional reward based on regime appropriateness."""
        regime_reward = 0
        
        if self.current_regime == 'bull':
            if action == 1:  # Buying in bull market
                regime_reward += 0.05
            elif action == 2 and actual_return > 0.02:  # Selling too early in bull
                regime_reward -= 0.1
        
        elif self.current_regime == 'bear':
            if action == 2:  # Selling in bear market
                regime_reward += 0.05
            elif action == 1 and actual_return < -0.02:  # Buying in bear market
                regime_reward -= 0.1
        
        # Neutral regime rewards conservative behavior
        elif self.current_regime == 'neutral':
            if action == 0:  # Holding in neutral market
                regime_reward += 0.02
        
        return regime_reward
    
    def get_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        if not self.trade_history:
            return self.performance_metrics
        
        # Total return
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        # Sharpe ratio (simplified)
        returns = [trade['actual_return'] for trade in self.trade_history if trade['executed']]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        portfolio_values = [self.initial_balance]
        for trade in self.trade_history:
            portfolio_values.append(trade.get('portfolio_value_before', portfolio_values[-1]))
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        profitable_trades = [t for t in self.trade_history if t['executed'] and t['reward'] > 0]
        executed_trades = [t for t in self.trade_history if t['executed']]
        win_rate = len(profitable_trades) / len(executed_trades) if executed_trades else 0
        
        # Noise resilience score (how well it avoids trading in noisy periods)
        noise_trades = [t for t in self.trade_history if t['executed'] and abs(t['actual_return']) < self.noise_level]
        noise_resilience_score = 1 - (len(noise_trades) / len(executed_trades)) if executed_trades else 1
        
        self.performance_metrics.update({
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'noise_resilience_score': noise_resilience_score
        })
        
        return self.performance_metrics

class MultiAssetTradingEnvironment(TradingEnvironment):
    """Extended environment for multiple asset trading."""
    
    def __init__(self, data_dict: Dict[str, pd.DataFrame], **kwargs):
        self.data_dict = data_dict
        self.assets = list(data_dict.keys())
        
        # Use first asset's data for base initialization
        super().__init__(data_dict[self.assets[0]], **kwargs)
        
        # Override action space for multiple assets
        # Actions: [hold, buy_asset1, sell_asset1, buy_asset2, sell_asset2, ...]
        self.action_space = spaces.Discrete(1 + 2 * len(self.assets))
        
        # Track positions for each asset
        self.positions = {asset: 0 for asset in self.assets}
    
    def _get_observation(self):
        """Get observation including all assets."""
        if self.current_step >= min(len(data) for data in self.data_dict.values()):
            return np.zeros(self.observation_space.shape[0])
        
        # Combine features from all assets
        all_features = []
        for asset in self.assets:
            if self.current_step < len(self.data_dict[asset]):
                asset_features = self.data_dict[asset].iloc[self.current_step].values
                all_features.extend(asset_features)
        
        # Portfolio state
        total_position_value = sum(
            self.positions[asset] * self._get_current_price(asset) 
            for asset in self.assets
        )
        
        portfolio_features = np.array([
            self.balance / self.initial_balance,
            total_position_value / self.initial_balance,
            len([p for p in self.positions.values() if p != 0])  # Number of positions
        ])
        
        observation = np.concatenate([all_features, portfolio_features])
        return np.nan_to_num(observation, nan=0.0).astype(np.float32)
    
    def _get_current_price(self, asset):
        """Get current price for specific asset."""
        if self.current_step >= len(self.data_dict[asset]):
            return self.data_dict[asset]['Close'].iloc[-1]
        return self.data_dict[asset]['Close'].iloc[self.current_step]

# Usage example
if __name__ == '__main__':
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100)
    sample_data = pd.DataFrame({
        'Close': np.cumsum(np.random.randn(100)) + 100,
        'Volume': np.random.randint(1000, 10000, 100),
        'returns': np.random.randn(100) * 0.02,
        'volatility': np.random.rand(100) * 0.05
    }, index=dates)
    
    # Test environment
    env = TradingEnvironment(sample_data, noise_level=0.01)
    
    print("Testing trading environment...")
    obs = env.reset()
    total_reward = 0
    
    for i in range(50):
        action = np.random.randint(0, 3)  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    metrics = env.get_performance_metrics()
    print(f"Episode completed. Total reward: {total_reward:.4f}")
    print(f"Performance metrics: {metrics}")