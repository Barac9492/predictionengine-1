# 🤖 Genius Prediction Engine v2.0

> An advanced AI-powered stock prediction system with noise-resilient trading guidance, self-learning mechanisms, and comprehensive backtesting capabilities.

## 🎯 Key Features

### Core Capabilities
- **Probabilistic Predictions**: Ensemble modeling with uncertainty quantification using Bayesian neural networks, XGBoost, and online learning
- **Noise-Resilient Trading Guidance**: Buy/Hold/Sell decisions with volatility-adjusted thresholds and confidence gating
- **Automated Proxy Discovery**: AI-driven discovery of indirect predictive indicators using business model analysis, supply chain insights, and social sentiment
- **Self-Learning & Adaptation**: Reinforcement learning optimization with curriculum learning and adaptive threshold adjustment
- **Real-time Monitoring**: Comprehensive dashboard with performance tracking and system health monitoring
- **Backtesting Framework**: Monte Carlo simulations with noise injection for robustness testing

### Architecture Overview

```
[Public Data Sources] --> [Proxy Discovery] --> [Feature Engineering] --> [ML Models] --> [Probabilistic Predictions] --> [Noise-Adjusted Guidance: Buy/Hold/Sell]
                          ^                                                                                                      |
                          |                                                                                                      v
                       [Feedback Loop: Monitor Trade Outcomes --> Retrain/Adapt Thresholds --> Enhance Proxies & Decisions]
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd predictionengine-1
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run a quick prediction**
```bash
python app/main_engine.py --stock AAPL
```

### Basic Usage

**Single Stock Prediction:**
```bash
python app/main_engine.py --stock TSLA
```

**Batch Predictions:**
```bash
python app/main_engine.py --batch
```

**Interactive Mode:**
```bash
python app/main_engine.py
> predict AAPL
> batch
> status
> quit
```

## 📊 Example Output

```json
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
    "rationale": "Strong upward signal (2.5%) above noise-adjusted threshold (2.2%), confidence: 85%",
    "metrics": {
      "thresh_buy": 0.022,
      "thresh_sell": -0.022,
      "confidence": 0.85
    }
  },
  "metadata": {
    "rl_enhanced": true,
    "data_points": 1250,
    "prediction_window": 30
  }
}
```

## 🏗️ Architecture Components

### 1. Data Ingestion (`features/data_ingestion/`)
- **Enhanced noise filtering** with outlier detection and multiple smoothing methods
- **Multi-source integration** (yfinance, Google Trends, economic indicators)
- **Quality assessment** with correlation-based proxy ranking
- **Feature engineering** with technical indicators and momentum features

### 2. Proxy Discovery (`features/proxy_discovery/`)
- **Automated discovery** using business model analysis and supply chain insights
- **Multi-strategy approach** covering competitors, news trends, and economic indicators
- **Quality scoring** and deduplication with confidence ranking
- **Dynamic proxy replacement** when performance degrades

### 3. Modeling & Prediction (`features/modeling/`)
- **Ensemble approach** combining Bayesian neural networks, XGBoost, and online learning
- **Probabilistic outputs** with uncertainty quantification
- **Sequential modeling** for time series with proper feature scaling
- **Adaptive training** with drift detection and automatic retraining

### 4. Trading Decision (`features/trading_decision/`)
- **Noise-adjusted thresholds** based on market volatility
- **Confidence gating** to avoid low-conviction trades
- **Market regime awareness** with bull/bear/neutral detection
- **Performance tracking** with accuracy scoring and false positive monitoring

### 5. Reinforcement Learning (`features/rl_training/`)
- **Trading environment** with noise injection and regime simulation
- **Adaptive RL trainer** with curriculum learning and self-enhancement
- **Multiple algorithms** (PPO, SAC, DQN) with automatic hyperparameter tuning
- **Performance callbacks** for real-time training monitoring

### 6. Monitoring & Feedback (`features/monitoring/`)
- **Real-time dashboard** with Streamlit interface
- **Feedback loop system** for automatic error detection and correction
- **Performance analytics** with comprehensive metrics tracking
- **Self-healing mechanisms** with automatic proxy replacement and threshold adjustment

### 7. Testing & Backtesting (`tests/`)
- **Walk-forward backtesting** with realistic transaction costs and slippage
- **Monte Carlo simulations** with noise injection for robustness testing
- **Performance metrics** including Sharpe ratio, Calmar ratio, and noise resilience scoring
- **Strategy comparison** framework for A/B testing different approaches

## 🎛️ Configuration

Edit `config.json` to customize behavior:

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

## 🔬 Advanced Features

### Reinforcement Learning Training
Train an RL agent for enhanced decision making:
```bash
python app/main_engine.py --train-rl TSLA
```

### Proxy Discovery
Discover new predictive indicators:
```bash
python app/main_engine.py --discover-proxies AAPL
```

### Backtesting
Run comprehensive backtests:
```python
from tests.backtesting_framework import BacktestingFramework

backtester = BacktestingFramework(initial_capital=10000)
result = backtester.run_backtest(
    stock='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01'
)
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
```

### Monte Carlo Robustness Testing
```python
mc_results = backtester.run_monte_carlo_backtest(
    stock='TSLA',
    n_simulations=100,
    noise_levels=[0.0, 0.01, 0.02, 0.03]
)
print(f"Noise Resilience Score: {mc_results['summary_stats']['noise_resilience_score']:.3f}")
```

### Monitoring Dashboard
Launch the real-time monitoring dashboard:
```bash
streamlit run features/monitoring/dashboard.py
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Accuracy Metrics**: Overall prediction accuracy, action-specific accuracy
- **Risk Metrics**: Sharpe ratio, maximum drawdown, Calmar ratio
- **Trading Metrics**: Win rate, profit factor, average trade duration
- **Noise Resilience**: Performance degradation under different noise levels
- **Confidence Calibration**: Correlation between predicted and actual confidence

## 🛠️ Development & Testing

### Running Tests
```bash
# Run backtesting framework
python tests/backtesting_framework.py

# Test individual components
python features/data_ingestion/pipeline.py
python features/modeling/predictor.py
python features/proxy_discovery/discovery.py
```

### Monitoring System Health
```bash
python app/main_engine.py --status
```

## 🎯 Implementation Strategy

### Phase 1: Core Infrastructure ✅
- [x] Data ingestion with noise filtering
- [x] Proxy discovery system
- [x] Ensemble modeling with uncertainty quantification
- [x] Trading guidance with volatility adjustment

### Phase 2: Self-Learning ✅
- [x] Reinforcement learning training environment
- [x] Adaptive threshold adjustment
- [x] Performance monitoring and feedback loops
- [x] Automatic model retraining

### Phase 3: Testing & Validation ✅
- [x] Comprehensive backtesting framework
- [x] Monte Carlo simulations with noise injection
- [x] Strategy comparison and optimization
- [x] Real-time monitoring dashboard

### Phase 4: Deployment & Scaling (Next Steps)
- [ ] Paper trading integration with Alpaca API
- [ ] Live trading with risk management
- [ ] Multi-asset portfolio optimization
- [ ] Cloud deployment with auto-scaling

## ⚠️ Risk Disclaimers

1. **Not Financial Advice**: This system is experimental and for educational purposes only
2. **Past Performance**: Historical results do not guarantee future performance
3. **Market Risk**: All trading involves risk of loss
4. **System Risk**: Automated systems can fail or behave unexpectedly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern ML frameworks: PyTorch, Pyro, XGBoost, stable-baselines3
- Data sources: yfinance, Google Trends, Federal Reserve Economic Data
- Inspired by quantitative finance research and behavioral economics

---

**Happy Trading! 📈🚀**

*Remember: Always do your own research and never invest more than you can afford to lose.*