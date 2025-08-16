# Pull Request

## Description
Brief description of changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Code refactoring

## Component Areas
- [ ] Data Ingestion (`features/data_ingestion/`)
- [ ] Modeling (`features/modeling/`)
- [ ] Trading Decision (`features/trading_decision/`)
- [ ] Proxy Discovery (`features/proxy_discovery/`)
- [ ] RL Training (`features/rl_training/`)
- [ ] Monitoring (`features/monitoring/`)
- [ ] Testing (`tests/`)
- [ ] Main Engine (`app/`)
- [ ] Configuration (`shared/`)
- [ ] Documentation

## Testing
- [ ] I have tested this change locally
- [ ] I have run the test suite (`python test_engine.py`)
- [ ] I have tested with multiple stocks
- [ ] I have tested edge cases
- [ ] I have added new tests for this feature

## Performance Impact
- [ ] No performance impact
- [ ] Minor performance improvement (<5%)
- [ ] Significant performance improvement (>5%)
- [ ] Performance regression (explain below)

**Performance Details:**
```
Add performance testing results here if applicable
```

## Backtesting Results
If this change affects trading logic, please include backtesting results:

**Before:**
- Total Return: X%
- Sharpe Ratio: X.XX
- Max Drawdown: X%

**After:**
- Total Return: X%
- Sharpe Ratio: X.XX
- Max Drawdown: X%

## Risk Assessment
- [ ] Low risk (documentation, minor bug fixes)
- [ ] Medium risk (new features, moderate changes)
- [ ] High risk (breaking changes, core algorithm changes)

**Risk Mitigation:**
- Describe any steps taken to mitigate risks
- Include fallback plans if applicable

## Configuration Changes
- [ ] No configuration changes required
- [ ] New optional configuration parameters
- [ ] Breaking configuration changes (update migration guide)

## Documentation
- [ ] README updated
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] Examples updated

## Checklist
- [ ] My code follows the project's coding standards
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes

## Related Issues
Closes #(issue number)
Related to #(issue number)

## Additional Notes
Any additional information that reviewers should know.