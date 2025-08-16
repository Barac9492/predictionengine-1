# Contributing to Genius Prediction Engine

Thank you for your interest in contributing to the Genius Prediction Engine! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### 1. Getting Started
- Fork the repository
- Clone your fork locally
- Create a new branch for your feature/fix
- Set up the development environment

### 2. Development Setup
```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/predictionengine-1.git
cd predictionengine-1

# Install dependencies
pip install -r requirements.txt

# Run tests to ensure everything works
python test_engine.py
```

### 3. Making Changes

#### Code Style
- Follow PEP 8 conventions
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Keep functions focused and modular

#### Testing
- Write tests for new features
- Ensure all existing tests pass
- Test with multiple stocks and time periods
- Include edge case testing

#### Performance
- Profile performance-critical code
- Include backtesting results for trading logic changes
- Consider memory usage for large datasets

### 4. Submission Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   python test_engine.py
   python app/main_engine.py --stock AAPL  # Test prediction
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 📋 Contribution Guidelines

### What We're Looking For
- **Bug Fixes**: Clear bug reports with reproduction steps
- **Performance Improvements**: Measurable performance gains
- **New Features**: Well-designed features that enhance prediction accuracy
- **Documentation**: Improvements to documentation and examples
- **Testing**: Additional test coverage and edge cases

### Priority Areas
1. **Data Sources**: Integration with new financial data APIs
2. **Prediction Models**: Advanced ML/AI models for better accuracy
3. **Risk Management**: Enhanced risk assessment and position sizing
4. **Real-time Trading**: Integration with trading platforms
5. **Visualization**: Better charts and analytics
6. **Mobile Support**: Mobile-friendly interfaces

### Code Review Process
1. All PRs require review from maintainers
2. Automated tests must pass
3. Performance regression tests for core changes
4. Documentation updates for user-facing changes

## 🏗️ Architecture Guidelines

### Module Structure
```
features/
├── data_ingestion/     # Data fetching and preprocessing
├── modeling/          # ML models and predictions
├── trading_decision/  # Trading logic and guidance
├── proxy_discovery/   # Automated proxy finding
├── rl_training/      # Reinforcement learning
└── monitoring/       # Performance tracking
```

### Design Principles
1. **Modularity**: Each component should be independent
2. **Testability**: All code should be easily testable
3. **Configurability**: Use configuration files, not hardcoded values
4. **Robustness**: Handle errors gracefully
5. **Performance**: Consider computational efficiency
6. **Documentation**: Code should be self-documenting

### Adding New Features

#### New Data Sources
```python
# features/data_ingestion/sources/your_source.py
class YourDataSource:
    def fetch_data(self, symbol, start_date, end_date):
        # Implementation
        pass
    
    def get_quality_score(self, data):
        # Data quality assessment
        pass
```

#### New Prediction Models
```python
# features/modeling/models/your_model.py
class YourModel:
    def __init__(self, config):
        self.config = config
    
    def fit(self, X, y):
        # Training logic
        pass
    
    def predict(self, X):
        # Return (prediction, uncertainty)
        pass
```

#### New Trading Strategies
```python
# features/trading_decision/strategies/your_strategy.py
class YourStrategy:
    def get_signal(self, prediction, confidence, market_context):
        # Return action, rationale, metrics
        pass
```

## 🧪 Testing Guidelines

### Test Categories
1. **Unit Tests**: Test individual functions/classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Measure execution time and memory
4. **Backtesting**: Validate trading strategies historically

### Writing Tests
```python
def test_your_feature():
    # Arrange
    engine = GeniusPredictionEngine(enable_rl=False, enable_feedback=False)
    
    # Act
    result = engine.get_prediction('AAPL')
    
    # Assert
    assert 'guidance' in result
    assert result['guidance']['action'] in ['Buy', 'Hold', 'Sell']
    
    # Cleanup
    engine.shutdown()
```

### Backtesting Requirements
For trading logic changes, include:
- Before/after performance metrics
- Multiple stock symbols tested
- Different market conditions (bull/bear/neutral)
- Risk metrics (Sharpe ratio, max drawdown)

## 📖 Documentation

### Required Documentation
- **Function Docstrings**: All public functions
- **Class Documentation**: Purpose and usage examples
- **Configuration**: New config parameters
- **API Changes**: Breaking changes in CHANGELOG.md

### Documentation Format
```python
def your_function(param1: str, param2: float) -> Dict:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    
    Returns:
        Dict containing result with keys:
        - 'success': bool indicating success
        - 'data': Any relevant data
    
    Raises:
        ValueError: When param1 is invalid
        
    Example:
        >>> result = your_function("test", 1.0)
        >>> print(result['success'])
        True
    """
```

## 🐛 Bug Reports

### Good Bug Reports Include
1. **Clear Title**: Concise description of the issue
2. **Reproduction Steps**: Step-by-step instructions
3. **Expected vs Actual**: What should happen vs what happens
4. **Environment**: OS, Python version, dependencies
5. **Logs**: Relevant error messages and stack traces
6. **Context**: Stock symbols, time periods, configuration

### Bug Report Template
Use the GitHub issue template for bug reports.

## 🚀 Feature Requests

### Good Feature Requests Include
1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Use Cases**: When would this be used?
4. **Implementation Ideas**: Technical approach (optional)
5. **Acceptance Criteria**: How to know it's complete

## 📊 Performance Standards

### Benchmarks
- **Import Time**: < 5 seconds for core modules
- **Prediction Time**: < 30 seconds per stock
- **Memory Usage**: < 2GB for normal operations
- **Accuracy**: Maintain or improve existing metrics

### Performance Testing
```bash
# Run performance tests
python -m cProfile -o profile.stats app/main_engine.py --stock AAPL
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

## 🏆 Recognition

Contributors will be:
- Listed in the project contributors
- Mentioned in release notes for significant contributions
- Invited to join the core team for outstanding contributions

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Request review from maintainers

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Genius Prediction Engine! 🚀📈