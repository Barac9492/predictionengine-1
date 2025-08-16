# app/main_engine.py
"""
Main orchestration engine for the Genius Prediction Engine.
Integrates all components and provides unified interface for trading guidance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from features.data_ingestion.pipeline import build_dataset
from features.modeling.predictor import EnsemblePredictor
from features.trading_decision.guidance import TradingGuidanceEngine, AdvancedGuidanceEngine
from features.proxy_discovery.discovery import ProxyDiscoveryEngine
from features.rl_training.trainer import AdaptiveRLTrainer, RLEnhancedGuidanceEngine
from features.monitoring.feedback_loop import FeedbackLoop
from features.monitoring.dashboard import PredictionEngineMonitor
from shared.config.targets import TARGET_STOCKS

class GeniusPredictionEngine:
    """
    Main orchestration class that integrates all components of the prediction engine.
    Provides unified interface for generating trading guidance with self-* mechanisms.
    """
    
    def __init__(self, 
                 config_path: str = "config.json",
                 enable_rl: bool = True,
                 enable_feedback: bool = True,
                 log_level: str = "INFO"):
        
        # Setup logging
        self.logger = self._setup_logging(log_level)
        self.logger.info("Initializing Genius Prediction Engine...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.guidance_engine = TradingGuidanceEngine()
        self.proxy_discovery = ProxyDiscoveryEngine()
        self.monitor = PredictionEngineMonitor()
        
        # Initialize predictors cache
        self.predictors = {}
        self.last_training = {}
        
        # RL components (optional)
        self.rl_trainer = None
        self.rl_enhanced_engine = None
        if enable_rl:
            self._initialize_rl_components()
        
        # Feedback loop (optional)
        self.feedback_loop = None
        if enable_feedback:
            self._initialize_feedback_loop()
        
        # Performance tracking
        self.prediction_history = []
        self.performance_metrics = {}
        
        self.logger.info("Genius Prediction Engine initialized successfully")
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('prediction_engine.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('GeniusPredictionEngine')
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file or create default."""
        default_config = {
            "stocks": TARGET_STOCKS,
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
                "enable_pytrends": True,
                "enable_news_sentiment": False,
                "enable_reddit": False
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return {**default_config, **config}  # Merge with defaults
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}. Using defaults.")
        else:
            self.logger.info("Config file not found. Creating default configuration.")
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _initialize_rl_components(self):
        """Initialize reinforcement learning components."""
        try:
            rl_config = self.config.get("rl_training", {})
            self.rl_trainer = AdaptiveRLTrainer(
                algorithm=rl_config.get("algorithm", "PPO"),
                learning_rate=rl_config.get("learning_rate", 3e-4)
            )
            
            # Try to load existing RL model
            model_path = "models/rl_model.zip"
            if os.path.exists(model_path):
                self.rl_trainer.model = self.rl_trainer.__class__.load(model_path)
                self.logger.info("Loaded existing RL model")
            
            self.rl_enhanced_engine = RLEnhancedGuidanceEngine(self.rl_trainer)
            self.logger.info("RL components initialized")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize RL components: {e}")
            self.rl_trainer = None
            self.rl_enhanced_engine = None
    
    def _initialize_feedback_loop(self):
        """Initialize feedback loop for self-correction."""
        try:
            self.feedback_loop = FeedbackLoop()
            self.feedback_loop.start_monitoring()
            self.logger.info("Feedback loop initialized and monitoring started")
        except Exception as e:
            self.logger.warning(f"Failed to initialize feedback loop: {e}")
            self.feedback_loop = None
    
    def get_prediction(self, 
                      stock: str,
                      force_retrain: bool = False,
                      use_rl: bool = True) -> Dict:
        """
        Generate prediction and trading guidance for a stock.
        
        Args:
            stock: Stock symbol to predict
            force_retrain: Force model retraining
            use_rl: Use RL-enhanced guidance if available
            
        Returns:
            Dict containing prediction, guidance, and metadata
        """
        self.logger.info(f"Generating prediction for {stock}")
        
        try:
            # Build dataset
            data = build_dataset(stock)
            if data.empty:
                return self._create_error_response(f"No data available for {stock}")
            
            # Check if we need to train/retrain predictor
            needs_training = (
                stock not in self.predictors or
                force_retrain or
                self._should_retrain(stock)
            )
            
            if needs_training:
                self.logger.info(f"Training predictor for {stock}")
                predictor = EnsemblePredictor(data, target_col='returns')
                self.predictors[stock] = predictor
                self.last_training[stock] = datetime.now()
            else:
                predictor = self.predictors[stock]
            
            # Generate prediction
            prediction_data = data.iloc[-self.config["prediction_window"]:]
            pred_mean, pred_std = predictor.predict(prediction_data)
            
            confidence = 1 / (1 + pred_std) if pred_std > 0 else 0.5
            volatility = data['volatility'].iloc[-1] if 'volatility' in data else 0.02
            
            # Get trading guidance
            if use_rl and self.rl_enhanced_engine:
                # Use RL-enhanced guidance
                observation = self._create_observation(data, pred_mean, confidence, volatility)
                action, rationale, metrics = self.rl_enhanced_engine.get_enhanced_guidance(
                    observation, pred_mean, confidence, volatility
                )
            else:
                # Use rule-based guidance
                action, rationale, metrics = self.guidance_engine.get_guidance(
                    pred_mean, confidence, volatility
                )
            
            # Create response
            response = {
                'timestamp': datetime.now().isoformat(),
                'stock': stock,
                'prediction': {
                    'expected_change': pred_mean,
                    'uncertainty': pred_std,
                    'confidence': confidence,
                    'volatility': volatility
                },
                'guidance': {
                    'action': action,
                    'rationale': rationale,
                    'metrics': metrics
                },
                'metadata': {
                    'model_trained': self.last_training.get(stock),
                    'data_points': len(data),
                    'prediction_window': len(prediction_data),
                    'rl_enhanced': use_rl and self.rl_enhanced_engine is not None
                }
            }
            
            # Log prediction for monitoring
            self._log_prediction(response)
            
            # Update performance tracking
            self.prediction_history.append(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {stock}: {str(e)}")
            return self._create_error_response(f"Prediction failed: {str(e)}")
    
    def _create_observation(self, data: pd.DataFrame, pred_mean: float, confidence: float, volatility: float) -> np.ndarray:
        """Create observation vector for RL model."""
        # Use last few rows of scaled data
        recent_data = data.iloc[-10:]
        
        # Extract key features
        price_features = [
            recent_data['Close'].iloc[-1] / recent_data['Close'].iloc[0] - 1,  # Price change
            volatility,
            confidence,
            pred_mean
        ]
        
        # Add proxy features if available
        proxy_cols = [col for col in data.columns if col not in 
                     ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'returns', 'volatility']]
        
        for col in proxy_cols[:6]:  # Limit to 6 proxies
            if col in recent_data.columns:
                price_features.append(recent_data[col].iloc[-1] / (recent_data[col].mean() + 1e-8))
        
        # Pad to fixed size
        while len(price_features) < 10:
            price_features.append(0.0)
        
        return np.array(price_features[:10], dtype=np.float32)
    
    def _should_retrain(self, stock: str) -> bool:
        """Check if model should be retrained."""
        if stock not in self.last_training:
            return True
        
        hours_since_training = (datetime.now() - self.last_training[stock]).total_seconds() / 3600
        return hours_since_training >= self.config["retrain_frequency_hours"]
    
    def _create_error_response(self, error_message: str) -> Dict:
        """Create standardized error response."""
        return {
            'timestamp': datetime.now().isoformat(),
            'error': True,
            'message': error_message,
            'guidance': {
                'action': 'Hold',
                'rationale': 'Error in prediction - defaulting to hold',
                'metrics': {}
            }
        }
    
    def _log_prediction(self, response: Dict):
        """Log prediction to monitoring system."""
        if self.monitor:
            try:
                pred = response['prediction']
                guidance = response['guidance']
                
                self.monitor.log_prediction(
                    stock=response['stock'],
                    predicted_change=pred['expected_change'],
                    confidence=pred['confidence'],
                    volatility=pred['volatility'],
                    action=guidance['action'],
                    rationale=guidance['rationale']
                )
            except Exception as e:
                self.logger.warning(f"Failed to log prediction: {e}")
    
    def batch_predictions(self, stocks: List[str] = None) -> Dict[str, Dict]:
        """Generate predictions for multiple stocks."""
        if stocks is None:
            stocks = self.config["stocks"]
        
        self.logger.info(f"Generating batch predictions for {len(stocks)} stocks")
        
        results = {}
        for stock in stocks:
            try:
                results[stock] = self.get_prediction(stock)
            except Exception as e:
                self.logger.error(f"Batch prediction failed for {stock}: {e}")
                results[stock] = self._create_error_response(f"Batch prediction failed: {e}")
        
        return results
    
    def train_rl_model(self, 
                      stock: str,
                      timesteps: int = None,
                      save_model: bool = True) -> Dict:
        """Train RL model for enhanced decision making."""
        if not self.rl_trainer:
            return {'error': 'RL trainer not initialized'}
        
        try:
            self.logger.info(f"Training RL model for {stock}")
            
            # Build training data
            data = build_dataset(stock)
            if data.empty:
                return {'error': f'No training data for {stock}'}
            
            # Use configured timesteps or default
            if timesteps is None:
                timesteps = self.config["rl_training"]["total_timesteps"]
            
            # Train model
            results = self.rl_trainer.train(
                data=data,
                total_timesteps=timesteps,
                save_path="models/"
            )
            
            # Save model if requested
            if save_model:
                model_path = f"models/rl_model_{stock}.zip"
                self.rl_trainer.model.save(model_path)
                results['model_path'] = model_path
            
            self.logger.info(f"RL training completed for {stock}")
            return results
            
        except Exception as e:
            self.logger.error(f"RL training failed for {stock}: {e}")
            return {'error': f'RL training failed: {e}'}
    
    def discover_new_proxies(self, stock: str) -> Dict:
        """Discover new proxy indicators for a stock."""
        try:
            self.logger.info(f"Discovering new proxies for {stock}")
            
            proxies = self.proxy_discovery.discover_proxies_for_stock(stock)
            
            # Export to configuration
            config_file = self.proxy_discovery.export_proxy_config(stock)
            
            return {
                'success': True,
                'stock': stock,
                'proxies_found': len(proxies),
                'top_proxies': [
                    {
                        'name': p.name,
                        'description': p.description,
                        'confidence': p.confidence
                    } for p in proxies[:5]
                ],
                'config_exported': config_file
            }
            
        except Exception as e:
            self.logger.error(f"Proxy discovery failed for {stock}: {e}")
            return {'error': f'Proxy discovery failed: {e}'}
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status report."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'prediction_engine': 'operational',
                'guidance_engine': 'operational',
                'proxy_discovery': 'operational',
                'monitoring': 'operational' if self.monitor else 'disabled',
                'rl_trainer': 'operational' if self.rl_trainer else 'disabled',
                'feedback_loop': 'operational' if self.feedback_loop else 'disabled'
            },
            'models_trained': len(self.predictors),
            'predictions_made': len(self.prediction_history),
            'configuration': {
                'stocks_tracked': len(self.config["stocks"]),
                'retrain_frequency': self.config["retrain_frequency_hours"],
                'rl_enabled': self.rl_trainer is not None
            }
        }
        
        # Add performance summary
        if self.prediction_history:
            recent_predictions = self.prediction_history[-10:]
            avg_confidence = np.mean([p['prediction']['confidence'] for p in recent_predictions])
            status['recent_performance'] = {
                'avg_confidence': avg_confidence,
                'predictions_last_hour': len([
                    p for p in recent_predictions 
                    if datetime.fromisoformat(p['timestamp']) > datetime.now() - timedelta(hours=1)
                ])
            }
        
        # Add feedback loop status
        if self.feedback_loop:
            try:
                health_report = self.feedback_loop.get_system_health_report()
                status['feedback_status'] = {
                    'monitoring_active': health_report['monitoring_status'] == 'active',
                    'queue_size': health_report['feedback_queue_size'],
                    'recent_actions': len(health_report['recent_actions'])
                }
            except:
                status['feedback_status'] = {'error': 'Unable to fetch feedback status'}
        
        return status
    
    def shutdown(self):
        """Gracefully shutdown the prediction engine."""
        self.logger.info("Shutting down Genius Prediction Engine...")
        
        # Stop feedback loop
        if self.feedback_loop:
            self.feedback_loop.stop_monitoring()
        
        # Save any trained models
        for stock, predictor in self.predictors.items():
            try:
                # Models are already saved during training
                pass
            except Exception as e:
                self.logger.warning(f"Failed to save model for {stock}: {e}")
        
        self.logger.info("Shutdown complete")

def main():
    """Main entry point for the prediction engine."""
    parser = argparse.ArgumentParser(description='Genius Prediction Engine')
    parser.add_argument('--stock', type=str, help='Stock to predict')
    parser.add_argument('--batch', action='store_true', help='Run batch predictions')
    parser.add_argument('--train-rl', type=str, help='Train RL model for stock')
    parser.add_argument('--discover-proxies', type=str, help='Discover proxies for stock')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--config', type=str, default='config.json', help='Config file path')
    parser.add_argument('--no-rl', action='store_true', help='Disable RL components')
    parser.add_argument('--no-feedback', action='store_true', help='Disable feedback loop')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = GeniusPredictionEngine(
        config_path=args.config,
        enable_rl=not args.no_rl,
        enable_feedback=not args.no_feedback
    )
    
    try:
        if args.status:
            # Show system status
            status = engine.get_system_status()
            print(json.dumps(status, indent=2))
        
        elif args.stock:
            # Single stock prediction
            result = engine.get_prediction(args.stock)
            print(json.dumps(result, indent=2))
        
        elif args.batch:
            # Batch predictions
            results = engine.batch_predictions()
            print(json.dumps(results, indent=2))
        
        elif args.train_rl:
            # Train RL model
            result = engine.train_rl_model(args.train_rl)
            print(json.dumps(result, indent=2))
        
        elif args.discover_proxies:
            # Discover proxies
            result = engine.discover_new_proxies(args.discover_proxies)
            print(json.dumps(result, indent=2))
        
        else:
            # Interactive mode
            print("Genius Prediction Engine - Interactive Mode")
            print("Commands: predict <STOCK>, batch, status, train-rl <STOCK>, quit")
            
            while True:
                try:
                    command = input("\n> ").strip().split()
                    
                    if not command:
                        continue
                    
                    if command[0] == 'quit':
                        break
                    elif command[0] == 'predict' and len(command) > 1:
                        result = engine.get_prediction(command[1].upper())
                        print(f"\nPrediction for {command[1].upper()}:")
                        print(f"Action: {result['guidance']['action']}")
                        print(f"Rationale: {result['guidance']['rationale']}")
                        if 'prediction' in result:
                            print(f"Expected Change: {result['prediction']['expected_change']:.2%}")
                            print(f"Confidence: {result['prediction']['confidence']:.1%}")
                    elif command[0] == 'batch':
                        print("Running batch predictions...")
                        results = engine.batch_predictions()
                        for stock, result in results.items():
                            action = result['guidance']['action']
                            print(f"{stock}: {action}")
                    elif command[0] == 'status':
                        status = engine.get_system_status()
                        print(f"System Status: {status['components']}")
                        print(f"Models Trained: {status['models_trained']}")
                        print(f"Predictions Made: {status['predictions_made']}")
                    else:
                        print("Unknown command. Available: predict <STOCK>, batch, status, quit")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
    
    finally:
        # Graceful shutdown
        engine.shutdown()

if __name__ == '__main__':
    main()