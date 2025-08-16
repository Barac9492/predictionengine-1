# features/trading_decision/guidance.py
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from shared.config.targets import VOL_ADJUST_THRESHOLD, CONFIDENCE_MIN

class TradingGuidanceEngine:
    """
    Enhanced trading guidance with noise-adjusted thresholds and confidence scoring.
    Implements buy/hold/sell logic with volatility-based threshold adjustment.
    """
    
    def __init__(self, base_threshold: float = VOL_ADJUST_THRESHOLD, min_confidence: float = CONFIDENCE_MIN):
        self.base_threshold = base_threshold
        self.min_confidence = min_confidence
        self.decision_history = []
        self.performance_metrics = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'false_positives': 0,
            'missed_opportunities': 0
        }
    
    def get_guidance(self, pred_change: float, confidence: float, volatility: float, 
                    additional_context: Dict[str, Any] = None) -> Tuple[str, str, Dict[str, float]]:
        """
        Generate trading guidance based on prediction, confidence, and market volatility.
        
        Args:
            pred_change: Predicted price change (as percentage)
            confidence: Model confidence score (0-1)
            volatility: Current volatility estimate
            additional_context: Optional dict with extra market context
        
        Returns:
            Tuple of (action, rationale, metrics)
        """
        # Adjust thresholds based on volatility (higher vol = wider bands)
        vol_multiplier = 1 + (volatility / 0.02)  # Scale based on typical 2% vol
        thresh_buy = self.base_threshold * vol_multiplier
        thresh_sell = -self.base_threshold * vol_multiplier
        
        # Additional context adjustments
        if additional_context:
            # Market regime adjustment (bull/bear)
            market_regime = additional_context.get('market_regime', 'neutral')
            if market_regime == 'bear':
                thresh_buy *= 1.5  # Be more conservative in bear markets
            elif market_regime == 'bull':
                thresh_sell *= 1.5  # Less eager to sell in bull markets
        
        metrics = {
            'thresh_buy': thresh_buy,
            'thresh_sell': thresh_sell,
            'confidence': confidence,
            'volatility': volatility,
            'pred_change': pred_change
        }
        
        # Decision logic with confidence gating
        if confidence >= self.min_confidence:
            if pred_change > thresh_buy:
                action = "Buy"
                rationale = f"Strong upward signal ({pred_change:.2%}) above noise-adjusted threshold ({thresh_buy:.2%}), confidence: {confidence:.1%}"
            elif pred_change < thresh_sell:
                action = "Sell" 
                rationale = f"Strong downward signal ({pred_change:.2%}) below noise-adjusted threshold ({thresh_sell:.2%}), confidence: {confidence:.1%}"
            else:
                action = "Hold"
                rationale = f"Signal within noise band ({thresh_sell:.2%} to {thresh_buy:.2%}), confidence: {confidence:.1%}"
        else:
            action = "Hold"
            rationale = f"Insufficient confidence ({confidence:.1%} < {self.min_confidence:.1%}) for action despite signal {pred_change:.2%}"
        
        # Log decision for performance tracking
        decision_record = {
            'timestamp': pd.Timestamp.now(),
            'action': action,
            'pred_change': pred_change,
            'confidence': confidence,
            'volatility': volatility,
            'rationale': rationale
        }
        self.decision_history.append(decision_record)
        
        return action, rationale, metrics
    
    def update_performance(self, actual_change: float, predicted_action: str) -> None:
        """Update performance metrics based on actual outcome."""
        self.performance_metrics['total_predictions'] += 1
        
        # Simple accuracy tracking
        if predicted_action == "Buy" and actual_change > 0:
            self.performance_metrics['correct_predictions'] += 1
        elif predicted_action == "Sell" and actual_change < 0:
            self.performance_metrics['correct_predictions'] += 1
        elif predicted_action == "Hold":
            # Hold is correct if change is within noise band
            if abs(actual_change) < self.base_threshold:
                self.performance_metrics['correct_predictions'] += 1
        else:
            # Track specific error types
            if predicted_action in ["Buy", "Sell"] and abs(actual_change) < self.base_threshold:
                self.performance_metrics['false_positives'] += 1
            elif predicted_action == "Hold" and abs(actual_change) > self.base_threshold:
                self.performance_metrics['missed_opportunities'] += 1
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get current performance metrics."""
        total = self.performance_metrics['total_predictions']
        if total == 0:
            return {'accuracy': 0.0, 'false_positive_rate': 0.0, 'miss_rate': 0.0}
        
        return {
            'accuracy': self.performance_metrics['correct_predictions'] / total,
            'false_positive_rate': self.performance_metrics['false_positives'] / total,
            'miss_rate': self.performance_metrics['missed_opportunities'] / total,
            'total_decisions': total
        }
    
    def adapt_thresholds(self, target_accuracy: float = 0.65) -> None:
        """Self-correcting mechanism to adjust thresholds based on performance."""
        perf = self.get_performance_summary()
        
        if perf['total_decisions'] < 10:  # Need minimum data
            return
        
        if perf['accuracy'] < target_accuracy:
            if perf['false_positive_rate'] > 0.2:
                # Too many false positives, increase thresholds
                self.base_threshold *= 1.1
                self.min_confidence = min(0.9, self.min_confidence * 1.05)
            elif perf['miss_rate'] > 0.3:
                # Missing too many opportunities, decrease thresholds
                self.base_threshold *= 0.95
                self.min_confidence = max(0.5, self.min_confidence * 0.95)

class AdvancedGuidanceEngine(TradingGuidanceEngine):
    """
    Extended guidance engine with ensemble voting and regime detection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regime_detector = MarketRegimeDetector()
        self.ensemble_weights = {'trend': 0.4, 'momentum': 0.3, 'volatility': 0.3}
    
    def get_ensemble_guidance(self, predictions: Dict[str, Tuple[float, float]], 
                            market_data: pd.DataFrame) -> Tuple[str, str, Dict[str, Any]]:
        """
        Get guidance from ensemble of different prediction models.
        
        Args:
            predictions: Dict of model_name -> (pred_change, confidence)
            market_data: Recent market data for regime detection
        """
        # Detect market regime
        regime = self.regime_detector.detect_regime(market_data)
        
        # Weighted ensemble of predictions
        weighted_pred = 0
        weighted_conf = 0
        total_weight = 0
        
        for model_name, (pred, conf) in predictions.items():
            weight = self.ensemble_weights.get(model_name, 1.0)
            weighted_pred += pred * conf * weight
            weighted_conf += conf * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred = weighted_pred / total_weight
            ensemble_conf = weighted_conf / total_weight
        else:
            ensemble_pred, ensemble_conf = 0, 0.5
        
        # Get volatility from market data
        volatility = market_data['Close'].pct_change().std() if len(market_data) > 1 else 0.02
        
        # Use base guidance with regime context
        return self.get_guidance(
            ensemble_pred, 
            ensemble_conf, 
            volatility,
            additional_context={'market_regime': regime, 'ensemble_details': predictions}
        )

class MarketRegimeDetector:
    """Simple market regime detection based on price trends and volatility."""
    
    def detect_regime(self, data: pd.DataFrame, lookback: int = 20) -> str:
        """Detect if market is in bull, bear, or neutral regime."""
        if len(data) < lookback:
            return 'neutral'
        
        recent_data = data.tail(lookback)
        price_change = (recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0]) - 1
        volatility = recent_data['Close'].pct_change().std()
        
        if price_change > 0.1 and volatility < 0.03:
            return 'bull'
        elif price_change < -0.1 and volatility > 0.05:
            return 'bear'
        else:
            return 'neutral'

# Usage example
if __name__ == '__main__':
    guidance_engine = TradingGuidanceEngine()
    
    # Example prediction
    action, rationale, metrics = guidance_engine.get_guidance(
        pred_change=0.045,  # 4.5% predicted increase
        confidence=0.75,    # 75% confidence
        volatility=0.025    # 2.5% volatility
    )
    
    print(f"Action: {action}")
    print(f"Rationale: {rationale}")
    print(f"Metrics: {metrics}")