# 📚 API Reference - Genius Prediction Engine

This document provides comprehensive API reference for the Genius Prediction Engine.

## 🚀 Main Engine API

### `GeniusPredictionEngine`

The main orchestration class that provides unified access to all prediction engine capabilities.

```python
from app.main_engine import GeniusPredictionEngine

engine = GeniusPredictionEngine(
    config_path="config.json",
    enable_rl=True,
    enable_feedback=True,
    log_level="INFO"
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | str | "config.json" | Path to configuration file |
| `enable_rl` | bool | True | Enable reinforcement learning components |
| `enable_feedback` | bool | True | Enable feedback loop monitoring |
| `log_level` | str | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |

#### Methods

##### `get_prediction(stock, force_retrain=False, use_rl=True)`

Generate prediction and trading guidance for a stock.

**Parameters:**
- `stock` (str): Stock symbol (e.g., "AAPL", "TSLA")
- `force_retrain` (bool): Force model retraining
- `use_rl` (bool): Use RL-enhanced guidance

**Returns:**
```python
{
    "timestamp": "2024-01-15T10:30:00",
    "stock": "AAPL",
    "prediction": {
        "expected_change": 0.025,
        "uncertainty": 0.015,
        "confidence": 0.85,
        "volatility": 0.022
    },
    "guidance": {
        "action": "Buy",
        "rationale": "Strong upward signal...",
        "metrics": {...}
    },
    "metadata": {
        "rl_enhanced": true,
        "data_points": 1250,
        "prediction_window": 30
    }
}
```

##### `batch_predictions(stocks=None)`

Generate predictions for multiple stocks.

**Parameters:**
- `stocks` (List[str], optional): List of stock symbols. Uses config stocks if None.

**Returns:**
```python
{
    "AAPL": {...},  # Individual prediction results
    "TSLA": {...},
    "NVDA": {...}
}
```

##### `get_system_status()`

Get comprehensive system status report.

**Returns:**
```python
{
    "timestamp": "2024-01-15T10:30:00",
    "components": {
        "prediction_engine": "operational",
        "guidance_engine": "operational",
        "rl_trainer": "operational"
    },
    "models_trained": 5,
    "predictions_made": 142,
    "recent_performance": {...}
}
```

##### `train_rl_model(stock, timesteps=None, save_model=True)`

Train reinforcement learning model for enhanced decision making.

**Parameters:**
- `stock` (str): Stock symbol for training
- `timesteps` (int, optional): Training timesteps. Uses config default if None.
- `save_model` (bool): Save trained model to disk

**Returns:**
```python
{
    "algorithm": "PPO",
    "total_timesteps": 100000,
    "final_metrics": {...},
    "model_path": "models/rl_model_AAPL.zip"
}
```

##### `discover_new_proxies(stock)`

Discover new proxy indicators for a stock.

**Returns:**
```python
{
    "success": true,
    "stock": "TSLA",
    "proxies_found": 12,
    "top_proxies": [
        {
            "name": "ev_sales_global",
            "description": "Global electric vehicle sales data",
            "confidence": 0.85
        }
    ],
    "config_exported": "discovered_proxies_TSLA_20240115.json"
}
```

---

## 📊 Data Ingestion API

### `build_dataset(stock, start_date='2021-01-01', end_date=None)`

Build comprehensive dataset for a stock including proxy data and features.

```python
from features.data_ingestion.pipeline import build_dataset

data = build_dataset('AAPL', start_date='2023-01-01')
```

**Parameters:**
- `stock` (str): Stock symbol
- `start_date` (str): Start date in 'YYYY-MM-DD' format
- `end_date` (str, optional): End date. Uses current date if None.

**Returns:**
- `pandas.DataFrame`: Comprehensive dataset with price data, proxies, and technical indicators

**Dataset Columns:**
- Price data: Open, High, Low, Close, Adj Close, Volume
- Returns: returns, volatility
- Technical indicators: ma_5, ma_20, rsi, bb_position, volume_ratio
- Proxy data: Various proxy indicators based on stock configuration

---

## 🧠 Modeling API

### `EnsemblePredictor`

Ensemble prediction model combining Bayesian neural networks, XGBoost, and online learning.

```python
from features.modeling.predictor import EnsemblePredictor

predictor = EnsemblePredictor(data, target_col='returns')
pred_mean, pred_std = predictor.predict(new_data)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | DataFrame | Required | Training dataset |
| `target_col` | str | 'returns' | Target column for prediction |

#### Methods

##### `predict(new_data, n_samples=100)`

Generate probabilistic predictions with uncertainty estimates.

**Parameters:**
- `new_data` (DataFrame): New data for prediction
- `n_samples` (int): Number of samples for Bayesian inference

**Returns:**
- `tuple`: (prediction_mean, prediction_std)

---

## 🎯 Trading Decision API

### `TradingGuidanceEngine`

Core trading guidance engine with noise-resilient decision making.

```python
from features.trading_decision.guidance import TradingGuidanceEngine

engine = TradingGuidanceEngine()
action, rationale, metrics = engine.get_guidance(
    pred_change=0.025,
    confidence=0.85,
    volatility=0.022
)
```

#### Methods

##### `get_guidance(pred_change, confidence, volatility, additional_context=None)`

Generate trading guidance based on prediction and market conditions.

**Parameters:**
- `pred_change` (float): Predicted price change (as percentage)
- `confidence` (float): Model confidence score (0-1)
- `volatility` (float): Current volatility estimate
- `additional_context` (dict, optional): Additional market context

**Returns:**
```python
(
    "Buy",  # Action: "Buy", "Hold", or "Sell"
    "Strong upward signal (2.5%) above noise-adjusted threshold (2.2%), confidence: 85%",  # Rationale
    {  # Metrics
        "thresh_buy": 0.022,
        "thresh_sell": -0.022,
        "confidence": 0.85,
        "volatility": 0.022,
        "pred_change": 0.025
    }
)
```

##### `update_performance(actual_change, predicted_action)`

Update performance metrics based on actual outcome.

**Parameters:**
- `actual_change` (float): Actual price change that occurred
- `predicted_action` (str): Previously predicted action

##### `get_performance_summary()`

Get current performance metrics.

**Returns:**
```python
{
    "accuracy": 0.68,
    "false_positive_rate": 0.15,
    "miss_rate": 0.12,
    "total_decisions": 150
}
```

---

## 🔍 Proxy Discovery API

### `ProxyDiscoveryEngine`

Automated system for discovering predictive proxy indicators.

```python
from features.proxy_discovery.discovery import ProxyDiscoveryEngine

discovery = ProxyDiscoveryEngine()
proxies = discovery.discover_proxies_for_stock('TSLA')
```

#### Methods

##### `discover_proxies_for_stock(stock, company_info=None)`

Discover proxy candidates for a given stock.

**Parameters:**
- `stock` (str): Stock symbol
- `company_info` (dict, optional): Company information for enhanced discovery

**Returns:**
- `List[ProxyCandidate]`: List of ranked proxy candidates

##### `get_top_proxies_for_stock(stock, limit=5)`

Get top performing proxies for a stock.

**Returns:**
```python
[
    {
        "name": "ev_sales_global",
        "description": "Global electric vehicle sales data",
        "source": "industry_reports",
        "keywords": ["electric vehicle sales"],
        "confidence": 0.85
    }
]
```

---

## 🤖 Reinforcement Learning API

### `AdaptiveRLTrainer`

Reinforcement learning trainer for optimizing trading decisions.

```python
from features.rl_training.trainer import AdaptiveRLTrainer

trainer = AdaptiveRLTrainer(algorithm='PPO')
results = trainer.train(data, total_timesteps=100000)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `algorithm` | str | 'PPO' | RL algorithm ('PPO', 'SAC', 'DQN') |
| `learning_rate` | float | 3e-4 | Learning rate for training |
| `batch_size` | int | 64 | Batch size for training |

#### Methods

##### `train(data, total_timesteps=100000, eval_freq=5000)`

Train the RL model on historical data.

**Parameters:**
- `data` (DataFrame): Training dataset
- `total_timesteps` (int): Total training timesteps
- `eval_freq` (int): Evaluation frequency

**Returns:**
```python
{
    "algorithm": "PPO",
    "total_timesteps": 100000,
    "final_metrics": {...},
    "training_history": [...],
    "model_path": "models/final_PPO_model.zip"
}
```

##### `predict_trading_action(observation, deterministic=True)`

Predict trading action for given observation.

**Returns:**
- `tuple`: (action_index, confidence)

---

## 📈 Backtesting API

### `BacktestingFramework`

Comprehensive backtesting system with Monte Carlo simulations.

```python
from tests.backtesting_framework import BacktestingFramework

backtester = BacktestingFramework(initial_capital=10000)
result = backtester.run_backtest(
    stock='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01'
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 10000 | Starting capital for backtesting |
| `transaction_cost` | float | 0.001 | Transaction cost as percentage |
| `slippage` | float | 0.0005 | Slippage as percentage |

#### Methods

##### `run_backtest(stock, start_date, end_date, prediction_window=30)`

Run comprehensive backtest for a stock.

**Returns:**
```python
BacktestResult(
    total_return=0.156,
    annualized_return=0.145,
    sharpe_ratio=1.23,
    max_drawdown=0.089,
    win_rate=0.62,
    total_trades=45,
    profit_factor=1.67
)
```

##### `run_monte_carlo_backtest(stock, n_simulations=100, noise_levels=[0.0, 0.01, 0.02])`

Run Monte Carlo simulations with noise injection.

**Returns:**
```python
{
    "simulations": [...],
    "noise_analysis": {
        0.0: {"mean_return": 0.15, "std_return": 0.08},
        0.01: {"mean_return": 0.12, "std_return": 0.12},
        0.02: {"mean_return": 0.08, "std_return": 0.15}
    },
    "summary_stats": {
        "noise_resilience_score": 0.78
    }
}
```

---

## 📊 Monitoring API

### `PredictionEngineMonitor`

Real-time monitoring and performance tracking system.

```python
from features.monitoring.dashboard import PredictionEngineMonitor

monitor = PredictionEngineMonitor()
```

#### Methods

##### `log_prediction(stock, predicted_change, confidence, volatility, action, rationale)`

Log a prediction to the monitoring database.

##### `get_performance_metrics(stock, days=30)`

Get performance metrics for a stock over specified period.

**Returns:**
```python
{
    "overall_accuracy": 0.68,
    "buy_accuracy": 0.72,
    "sell_accuracy": 0.65,
    "hold_accuracy": 0.71,
    "confidence_calibration": 0.82,
    "prediction_correlation": 0.34
}
```

---

## ⚙️ Configuration

### Configuration File Format

```json
{
  "stocks": ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
  "prediction_window": 30,
  "retrain_frequency_hours": 24,
  "confidence_threshold": 0.7,
  "volatility_threshold": 0.03,
  "rl_training": {
    "algorithm": "PPO",
    "learning_rate": 3e-4,
    "total_timesteps": 100000
  },
  "data_sources": {
    "enable_pytrends": true,
    "enable_news_sentiment": false,
    "enable_reddit": false
  }
}
```

### Environment Variables

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_key
FRED_API_KEY=your_key

# Database
DATABASE_URL=sqlite:///monitoring.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=prediction_engine.log
```

---

## 🚨 Error Handling

All API methods follow consistent error handling patterns:

### Exception Types

- `ValueError`: Invalid parameters or configuration
- `DataError`: Data fetching or processing issues
- `ModelError`: Model training or prediction failures
- `APIError`: External API communication problems

### Error Response Format

```python
{
    "error": true,
    "error_type": "ModelError",
    "message": "Insufficient training data for AAPL",
    "timestamp": "2024-01-15T10:30:00",
    "details": {...}
}
```

---

## 📝 Usage Examples

### Basic Prediction

```python
from app.main_engine import GeniusPredictionEngine

# Initialize engine
engine = GeniusPredictionEngine()

# Get prediction
result = engine.get_prediction('AAPL')
print(f"Action: {result['guidance']['action']}")
print(f"Confidence: {result['prediction']['confidence']:.1%}")

# Cleanup
engine.shutdown()
```

### Batch Processing

```python
# Get predictions for multiple stocks
results = engine.batch_predictions(['AAPL', 'TSLA', 'NVDA'])

for stock, result in results.items():
    if not result.get('error'):
        action = result['guidance']['action']
        confidence = result['prediction']['confidence']
        print(f"{stock}: {action} (confidence: {confidence:.1%})")
```

### Custom Backtesting

```python
from tests.backtesting_framework import BacktestingFramework

# Initialize backtester
backtester = BacktestingFramework(
    initial_capital=50000,
    transaction_cost=0.0005,
    slippage=0.0003
)

# Run backtest
result = backtester.run_backtest(
    stock='TSLA',
    start_date='2022-01-01',
    end_date='2023-12-31',
    prediction_window=20
)

# Display results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
```

---

*For more examples and tutorials, see the main README.md and notebooks/ directory.*