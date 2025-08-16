# features/monitoring/feedback_loop.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import time
from abc import ABC, abstractmethod

from features.data_ingestion.pipeline import build_dataset
from features.modeling.predictor import EnsemblePredictor
from features.trading_decision.guidance import TradingGuidanceEngine
from features.proxy_discovery.discovery import ProxyDiscoveryEngine

@dataclass
class FeedbackSignal:
    """Data class for feedback signals."""
    timestamp: datetime
    component: str
    signal_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    data: Dict
    action_required: bool = False

class FeedbackProcessor(ABC):
    """Abstract base class for feedback processors."""
    
    @abstractmethod
    def process_feedback(self, signal: FeedbackSignal) -> Dict:
        """Process a feedback signal and return actions to take."""
        pass

class PerformanceFeedbackProcessor(FeedbackProcessor):
    """Processes performance-related feedback and suggests improvements."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.performance_threshold = 0.6  # Minimum acceptable accuracy
        
    def process_feedback(self, signal: FeedbackSignal) -> Dict:
        """Process performance feedback signal."""
        actions = []
        
        if signal.signal_type == 'accuracy_drop':
            accuracy = signal.data.get('accuracy', 0)
            
            if accuracy < 0.4:
                actions.append({
                    'type': 'retrain_models',
                    'priority': 'high',
                    'params': {'full_retrain': True}
                })
                actions.append({
                    'type': 'discover_new_proxies',
                    'priority': 'medium',
                    'params': {'expand_search': True}
                })
            elif accuracy < self.performance_threshold:
                actions.append({
                    'type': 'tune_hyperparameters',
                    'priority': 'medium',
                    'params': {'method': 'bayesian_optimization'}
                })
                actions.append({
                    'type': 'update_proxy_weights',
                    'priority': 'low',
                    'params': {'correlation_threshold': 0.2}
                })
        
        elif signal.signal_type == 'prediction_drift':
            drift_score = signal.data.get('drift_score', 0)
            
            if drift_score > 0.3:
                actions.append({
                    'type': 'update_data_sources',
                    'priority': 'high',
                    'params': {'refresh_all': True}
                })
                actions.append({
                    'type': 'recalibrate_thresholds',
                    'priority': 'medium',
                    'params': {'adaptive': True}
                })
        
        return {'actions': actions, 'processed': True}

class DataQualityFeedbackProcessor(FeedbackProcessor):
    """Processes data quality feedback."""
    
    def process_feedback(self, signal: FeedbackSignal) -> Dict:
        """Process data quality feedback signal."""
        actions = []
        
        if signal.signal_type == 'missing_data':
            missing_percentage = signal.data.get('missing_percentage', 0)
            
            if missing_percentage > 0.3:
                actions.append({
                    'type': 'find_alternative_data_sources',
                    'priority': 'high',
                    'params': {'target_coverage': 0.9}
                })
            elif missing_percentage > 0.1:
                actions.append({
                    'type': 'improve_data_imputation',
                    'priority': 'medium',
                    'params': {'method': 'advanced_interpolation'}
                })
        
        elif signal.signal_type == 'data_staleness':
            staleness_hours = signal.data.get('staleness_hours', 0)
            
            if staleness_hours > 24:
                actions.append({
                    'type': 'refresh_data_pipeline',
                    'priority': 'high',
                    'params': {'force_update': True}
                })
        
        return {'actions': actions, 'processed': True}

class ProxyFeedbackProcessor(FeedbackProcessor):
    """Processes proxy-related feedback."""
    
    def process_feedback(self, signal: FeedbackSignal) -> Dict:
        """Process proxy feedback signal."""
        actions = []
        
        if signal.signal_type == 'proxy_degradation':
            correlation_drop = signal.data.get('correlation_drop', 0)
            proxy_name = signal.data.get('proxy_name', '')
            
            if correlation_drop > 0.3:
                actions.append({
                    'type': 'remove_proxy',
                    'priority': 'high',
                    'params': {'proxy_name': proxy_name}
                })
                actions.append({
                    'type': 'search_replacement_proxy',
                    'priority': 'medium',
                    'params': {'target_correlation': 0.5}
                })
            elif correlation_drop > 0.1:
                actions.append({
                    'type': 'adjust_proxy_weight',
                    'priority': 'low',
                    'params': {'proxy_name': proxy_name, 'adjustment': -0.2}
                })
        
        return {'actions': actions, 'processed': True}

class FeedbackLoop:
    """
    Self-correcting feedback loop system that monitors performance,
    detects issues, and automatically applies corrective actions.
    """
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.logger = self._setup_logging()
        
        # Feedback processors
        self.processors = {
            'performance': PerformanceFeedbackProcessor(db_path),
            'data_quality': DataQualityFeedbackProcessor(),
            'proxy': ProxyFeedbackProcessor()
        }
        
        # System components
        self.guidance_engine = TradingGuidanceEngine()
        self.proxy_discovery = ProxyDiscoveryEngine()
        
        # Feedback queue and processing
        self.feedback_queue = []
        self.processing_thread = None
        self.running = False
        
        # Monitoring state
        self.last_check = {}
        self.check_intervals = {
            'performance': timedelta(hours=1),
            'data_quality': timedelta(minutes=30),
            'proxy_health': timedelta(hours=6)
        }
        
        # Action history
        self.action_history = []
        
    def _setup_logging(self):
        """Setup logging for the feedback loop."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('feedback_loop.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('FeedbackLoop')
    
    def start_monitoring(self):
        """Start the continuous monitoring process."""
        self.running = True
        self.processing_thread = threading.Thread(target=self._monitoring_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Feedback loop monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.logger.info("Feedback loop monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check various system components
                self._check_performance()
                self._check_data_quality()
                self._check_proxy_health()
                
                # Process any queued feedback signals
                self._process_feedback_queue()
                
                # Sleep for a short interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(300)  # Wait 5 minutes before retrying
    
    def _check_performance(self):
        """Check system performance and generate feedback if needed."""
        if not self._should_check('performance'):
            return
        
        try:
            # Get recent performance metrics
            conn = sqlite3.connect(self.db_path)
            
            # Check accuracy over last 24 hours
            recent_query = '''
                SELECT AVG(accuracy_score) as avg_accuracy,
                       COUNT(*) as prediction_count
                FROM predictions 
                WHERE timestamp > datetime('now', '-24 hours')
                AND actual_change IS NOT NULL
            '''
            
            result = pd.read_sql_query(recent_query, conn)
            conn.close()
            
            if not result.empty and result.iloc[0]['prediction_count'] > 5:
                avg_accuracy = result.iloc[0]['avg_accuracy']
                
                if avg_accuracy < 0.6:
                    signal = FeedbackSignal(
                        timestamp=datetime.now(),
                        component='prediction_system',
                        signal_type='accuracy_drop',
                        severity='high' if avg_accuracy < 0.4 else 'medium',
                        message=f"Prediction accuracy dropped to {avg_accuracy:.2%}",
                        data={'accuracy': avg_accuracy},
                        action_required=True
                    )
                    self.add_feedback_signal(signal)
            
            self.last_check['performance'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Performance check failed: {str(e)}")
    
    def _check_data_quality(self):
        """Check data quality and completeness."""
        if not self._should_check('data_quality'):
            return
        
        try:
            # Check data freshness and completeness for key stocks
            stocks = ['AAPL', 'TSLA', 'NVDA']
            
            for stock in stocks:
                try:
                    data = build_dataset(stock)
                    
                    if data.empty:
                        signal = FeedbackSignal(
                            timestamp=datetime.now(),
                            component='data_ingestion',
                            signal_type='missing_data',
                            severity='critical',
                            message=f"No data available for {stock}",
                            data={'stock': stock, 'missing_percentage': 1.0},
                            action_required=True
                        )
                        self.add_feedback_signal(signal)
                        continue
                    
                    # Check for missing values
                    missing_percentage = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
                    
                    if missing_percentage > 0.1:
                        signal = FeedbackSignal(
                            timestamp=datetime.now(),
                            component='data_ingestion',
                            signal_type='missing_data',
                            severity='medium' if missing_percentage < 0.3 else 'high',
                            message=f"High missing data percentage for {stock}: {missing_percentage:.1%}",
                            data={'stock': stock, 'missing_percentage': missing_percentage},
                            action_required=missing_percentage > 0.2
                        )
                        self.add_feedback_signal(signal)
                    
                    # Check data staleness
                    last_update = data.index[-1] if not data.empty else datetime.now() - timedelta(days=7)
                    staleness_hours = (datetime.now() - last_update).total_seconds() / 3600
                    
                    if staleness_hours > 12:  # Data older than 12 hours
                        signal = FeedbackSignal(
                            timestamp=datetime.now(),
                            component='data_ingestion',
                            signal_type='data_staleness',
                            severity='medium' if staleness_hours < 24 else 'high',
                            message=f"Stale data for {stock}: {staleness_hours:.1f} hours old",
                            data={'stock': stock, 'staleness_hours': staleness_hours},
                            action_required=staleness_hours > 24
                        )
                        self.add_feedback_signal(signal)
                
                except Exception as e:
                    self.logger.warning(f"Data quality check failed for {stock}: {str(e)}")
            
            self.last_check['data_quality'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Data quality check failed: {str(e)}")
    
    def _check_proxy_health(self):
        """Check proxy performance and health."""
        if not self._should_check('proxy_health'):
            return
        
        try:
            # This would check proxy correlation degradation over time
            # For now, simulate some proxy health checks
            
            stocks = ['AAPL', 'TSLA']
            
            for stock in stocks:
                # Simulate proxy performance check
                proxies = self.proxy_discovery.get_top_proxies_for_stock(stock, limit=3)
                
                for proxy in proxies:
                    # Simulate correlation degradation
                    current_correlation = proxy.get('confidence', 0.5)
                    
                    if current_correlation < 0.3:
                        signal = FeedbackSignal(
                            timestamp=datetime.now(),
                            component='proxy_system',
                            signal_type='proxy_degradation',
                            severity='medium',
                            message=f"Proxy {proxy['name']} for {stock} showing low correlation",
                            data={
                                'stock': stock,
                                'proxy_name': proxy['name'],
                                'correlation_drop': 0.5 - current_correlation
                            },
                            action_required=True
                        )
                        self.add_feedback_signal(signal)
            
            self.last_check['proxy_health'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Proxy health check failed: {str(e)}")
    
    def _should_check(self, check_type: str) -> bool:
        """Determine if a particular check should be run."""
        last_check = self.last_check.get(check_type)
        interval = self.check_intervals.get(check_type, timedelta(hours=1))
        
        if last_check is None:
            return True
        
        return datetime.now() - last_check >= interval
    
    def add_feedback_signal(self, signal: FeedbackSignal):
        """Add a feedback signal to the processing queue."""
        self.feedback_queue.append(signal)
        self.logger.info(f"Added feedback signal: {signal.signal_type} - {signal.message}")
    
    def _process_feedback_queue(self):
        """Process all queued feedback signals."""
        while self.feedback_queue:
            signal = self.feedback_queue.pop(0)
            self._process_single_feedback(signal)
    
    def _process_single_feedback(self, signal: FeedbackSignal):
        """Process a single feedback signal."""
        try:
            # Determine which processor to use
            processor_map = {
                'prediction_system': 'performance',
                'data_ingestion': 'data_quality',
                'proxy_system': 'proxy'
            }
            
            processor_key = processor_map.get(signal.component)
            if processor_key and processor_key in self.processors:
                processor = self.processors[processor_key]
                result = processor.process_feedback(signal)
                
                if result.get('processed'):
                    actions = result.get('actions', [])
                    for action in actions:
                        self._execute_action(action, signal)
                
                self.logger.info(f"Processed feedback signal: {signal.signal_type}")
            else:
                self.logger.warning(f"No processor found for component: {signal.component}")
        
        except Exception as e:
            self.logger.error(f"Failed to process feedback signal: {str(e)}")
    
    def _execute_action(self, action: Dict, original_signal: FeedbackSignal):
        """Execute a corrective action."""
        try:
            action_type = action.get('type')
            params = action.get('params', {})
            priority = action.get('priority', 'medium')
            
            self.logger.info(f"Executing action: {action_type} (priority: {priority})")
            
            # Record action
            action_record = {
                'timestamp': datetime.now(),
                'action_type': action_type,
                'params': params,
                'priority': priority,
                'trigger_signal': original_signal.signal_type,
                'component': original_signal.component
            }
            
            # Execute based on action type
            if action_type == 'retrain_models':
                self._action_retrain_models(params)
            elif action_type == 'discover_new_proxies':
                self._action_discover_proxies(params)
            elif action_type == 'tune_hyperparameters':
                self._action_tune_hyperparameters(params)
            elif action_type == 'update_data_sources':
                self._action_update_data_sources(params)
            elif action_type == 'recalibrate_thresholds':
                self._action_recalibrate_thresholds(params)
            else:
                self.logger.warning(f"Unknown action type: {action_type}")
                return
            
            action_record['status'] = 'completed'
            action_record['completion_time'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to execute action {action_type}: {str(e)}")
            action_record['status'] = 'failed'
            action_record['error'] = str(e)
        
        finally:
            self.action_history.append(action_record)
    
    def _action_retrain_models(self, params: Dict):
        """Action: Retrain prediction models."""
        self.logger.info("Action: Retraining models")
        # This would trigger model retraining
        # For now, just log the action
        pass
    
    def _action_discover_proxies(self, params: Dict):
        """Action: Discover new proxy indicators."""
        self.logger.info("Action: Discovering new proxies")
        # This would trigger proxy discovery
        pass
    
    def _action_tune_hyperparameters(self, params: Dict):
        """Action: Tune model hyperparameters."""
        self.logger.info("Action: Tuning hyperparameters")
        # This would trigger hyperparameter optimization
        pass
    
    def _action_update_data_sources(self, params: Dict):
        """Action: Update data sources."""
        self.logger.info("Action: Updating data sources")
        # This would refresh data pipelines
        pass
    
    def _action_recalibrate_thresholds(self, params: Dict):
        """Action: Recalibrate decision thresholds."""
        self.logger.info("Action: Recalibrating thresholds")
        # This would adjust guidance thresholds
        adaptive = params.get('adaptive', False)
        if adaptive:
            # Implement adaptive threshold adjustment
            pass
    
    def get_system_health_report(self) -> Dict:
        """Generate a comprehensive system health report."""
        report = {
            'timestamp': datetime.now(),
            'monitoring_status': 'active' if self.running else 'inactive',
            'last_checks': self.last_check,
            'recent_actions': self.action_history[-10:],  # Last 10 actions
            'feedback_queue_size': len(self.feedback_queue),
            'system_components': {
                'data_ingestion': 'operational',
                'prediction_models': 'operational', 
                'guidance_engine': 'operational',
                'proxy_discovery': 'operational'
            }
        }
        
        # Add performance summary
        try:
            conn = sqlite3.connect(self.db_path)
            perf_query = '''
                SELECT AVG(accuracy_score) as avg_accuracy,
                       COUNT(*) as total_predictions
                FROM predictions 
                WHERE timestamp > datetime('now', '-7 days')
                AND actual_change IS NOT NULL
            '''
            perf_result = pd.read_sql_query(perf_query, conn)
            conn.close()
            
            if not perf_result.empty:
                report['performance_summary'] = {
                    'avg_accuracy_7d': perf_result.iloc[0]['avg_accuracy'],
                    'total_predictions_7d': perf_result.iloc[0]['total_predictions']
                }
        except:
            report['performance_summary'] = {'error': 'Unable to fetch performance data'}
        
        return report

# Usage example
if __name__ == '__main__':
    # Initialize feedback loop
    feedback_loop = FeedbackLoop()
    
    # Start monitoring
    feedback_loop.start_monitoring()
    
    # Let it run for a bit
    print("Feedback loop started. Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(10)
            
            # Print system health report every minute
            if int(time.time()) % 60 == 0:
                report = feedback_loop.get_system_health_report()
                print(f"System Health: {report['monitoring_status']}, "
                      f"Queue: {report['feedback_queue_size']}, "
                      f"Actions: {len(report['recent_actions'])}")
    
    except KeyboardInterrupt:
        print("Stopping feedback loop...")
        feedback_loop.stop_monitoring()
        print("Feedback loop stopped.")