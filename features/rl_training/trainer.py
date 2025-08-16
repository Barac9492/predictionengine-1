# features/rl_training/trainer.py
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import pickle
from datetime import datetime
import os

from .environment import TradingEnvironment, MultiAssetTradingEnvironment
from features.trading_decision.guidance import TradingGuidanceEngine

class TradingPerformanceCallback(BaseCallback):
    """Custom callback to track trading performance during RL training."""
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super(TradingPerformanceCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.performance_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Get environment performance metrics
            if hasattr(self.training_env.envs[0], 'get_performance_metrics'):
                metrics = self.training_env.envs[0].get_performance_metrics()
                metrics['step'] = self.n_calls
                self.performance_history.append(metrics)
                
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                          f"Return={metrics.get('total_return', 0):.3f}")
        
        return True

class AdaptiveRLTrainer:
    """
    Adaptive RL trainer that learns to optimize trading decisions with noise resilience.
    Supports multiple algorithms, curriculum learning, and self-enhancement mechanisms.
    """
    
    def __init__(self, 
                 algorithm: str = 'PPO',
                 learning_rate: float = 3e-4,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 device: str = 'auto'):
        
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        
        self.model = None
        self.training_env = None
        self.eval_env = None
        
        # Training history
        self.training_history = []
        self.adaptation_history = []
        
        # Performance tracking
        self.best_performance = {
            'sharpe_ratio': -np.inf,
            'model_path': None,
            'parameters': None
        }
        
        # Self-enhancement parameters
        self.enhancement_enabled = True
        self.enhancement_frequency = 5000  # Steps between enhancements
        self.last_enhancement_step = 0
        
    def create_training_environment(self, 
                                  data: pd.DataFrame, 
                                  noise_levels: List[float] = [0.01, 0.02, 0.03],
                                  n_envs: int = 4) -> DummyVecEnv:
        """Create vectorized training environment with curriculum learning."""
        
        def make_env(noise_level):
            def _init():
                return TradingEnvironment(
                    data=data,
                    noise_level=noise_level,
                    regime_aware=True,
                    transaction_cost=0.001
                )
            return _init
        
        # Create environments with different noise levels for robustness
        env_fns = []
        for i in range(n_envs):
            noise_level = noise_levels[i % len(noise_levels)]
            env_fns.append(make_env(noise_level))
        
        return DummyVecEnv(env_fns)
    
    def initialize_model(self, env):
        """Initialize RL model based on algorithm choice."""
        
        common_params = {
            'env': env,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'device': self.device,
            'verbose': 1
        }
        
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                batch_size=self.batch_size,
                n_steps=2048,
                **common_params
            )
        elif self.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                buffer_size=100000,
                batch_size=self.batch_size,
                **common_params
            )
        elif self.algorithm == 'DQN':
            self.model = DQN(
                'MlpPolicy',
                buffer_size=100000,
                batch_size=self.batch_size,
                exploration_fraction=0.3,
                **common_params
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        print(f"Initialized {self.algorithm} model with {self.model.policy}")
    
    def train(self, 
              data: pd.DataFrame,
              total_timesteps: int = 100000,
              eval_freq: int = 5000,
              save_path: str = "models/") -> Dict[str, Any]:
        """Main training loop with adaptive enhancements."""
        
        print(f"Starting RL training with {self.algorithm} for {total_timesteps} timesteps...")
        
        # Create environments
        self.training_env = self.create_training_environment(data)
        self.eval_env = TradingEnvironment(data, noise_level=0.02)
        
        # Initialize model
        self.initialize_model(self.training_env)
        
        # Setup callbacks
        performance_callback = TradingPerformanceCallback(eval_freq=1000)
        eval_callback = EvalCallback(
            DummyVecEnv([lambda: self.eval_env]),
            eval_freq=eval_freq,
            best_model_save_path=save_path,
            log_path=save_path
        )
        
        callbacks = [performance_callback, eval_callback]
        
        # Training with adaptive enhancement
        steps_completed = 0
        enhancement_step = self.enhancement_frequency
        
        while steps_completed < total_timesteps:
            # Calculate steps for this iteration
            steps_this_iter = min(enhancement_step, total_timesteps - steps_completed)
            
            print(f"Training iteration: {steps_completed}-{steps_completed + steps_this_iter}")
            
            # Train model
            self.model.learn(
                total_timesteps=steps_this_iter,
                callback=callbacks,
                reset_num_timesteps=False
            )
            
            steps_completed += steps_this_iter
            
            # Self-enhancement mechanism
            if self.enhancement_enabled and steps_completed < total_timesteps:
                self._apply_self_enhancement(performance_callback.performance_history)
            
            # Update training history
            if performance_callback.performance_history:
                self.training_history.extend(performance_callback.performance_history)
                performance_callback.performance_history = []  # Reset for next iteration
        
        # Final evaluation
        final_metrics = self._evaluate_model(data)
        
        # Save final model
        final_model_path = os.path.join(save_path, f"final_{self.algorithm}_model.zip")
        self.model.save(final_model_path)
        
        training_results = {
            'algorithm': self.algorithm,
            'total_timesteps': total_timesteps,
            'final_metrics': final_metrics,
            'training_history': self.training_history,
            'adaptation_history': self.adaptation_history,
            'best_performance': self.best_performance,
            'model_path': final_model_path
        }
        
        return training_results
    
    def _apply_self_enhancement(self, recent_performance: List[Dict]) -> None:
        """Apply self-enhancement mechanisms based on recent performance."""
        
        if len(recent_performance) < 3:
            return
        
        # Analyze recent performance trends
        recent_metrics = recent_performance[-3:]
        sharpe_trend = [m.get('sharpe_ratio', 0) for m in recent_metrics]
        return_trend = [m.get('total_return', 0) for m in recent_metrics]
        
        # Check if performance is declining
        sharpe_declining = len(sharpe_trend) > 1 and sharpe_trend[-1] < sharpe_trend[0]
        return_declining = len(return_trend) > 1 and return_trend[-1] < return_trend[0]
        
        enhancement_applied = False
        
        # Enhancement 1: Adjust learning rate if performance declining
        if sharpe_declining and return_declining:
            old_lr = self.model.learning_rate
            if hasattr(self.model, 'lr_schedule'):
                # Reduce learning rate for fine-tuning
                new_lr = old_lr * 0.5
                self.model.lr_schedule = lambda _: new_lr
                print(f"Enhanced: Reduced learning rate from {old_lr} to {new_lr}")
                enhancement_applied = True
        
        # Enhancement 2: Increase exploration if stuck in local optimum
        avg_sharpe = np.mean(sharpe_trend)
        if avg_sharpe < 0.5 and self.algorithm == 'DQN':
            if hasattr(self.model, 'exploration_rate'):
                old_rate = self.model.exploration_rate
                new_rate = min(0.5, old_rate * 1.5)
                self.model.exploration_rate = new_rate
                print(f"Enhanced: Increased exploration rate from {old_rate} to {new_rate}")
                enhancement_applied = True
        
        # Enhancement 3: Curriculum learning - increase environment difficulty
        if avg_sharpe > 1.0:  # Good performance, increase challenge
            # This would involve modifying the environment parameters
            print("Enhanced: Performance good, considering curriculum advancement")
            enhancement_applied = True
        
        if enhancement_applied:
            self.adaptation_history.append({
                'step': self.last_enhancement_step,
                'enhancement_type': 'adaptive_parameters',
                'trigger_metrics': recent_metrics[-1],
                'timestamp': datetime.now()
            })
        
        self.last_enhancement_step = recent_metrics[-1].get('step', 0)
    
    def _evaluate_model(self, data: pd.DataFrame, n_episodes: int = 10) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        
        eval_env = TradingEnvironment(data, noise_level=0.02)
        
        episode_metrics = []
        
        for episode in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
            
            metrics = eval_env.get_performance_metrics()
            metrics['episode_reward'] = episode_reward
            episode_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in episode_metrics[0].keys():
            values = [m[key] for m in episode_metrics if key in m]
            aggregated_metrics[f'mean_{key}'] = np.mean(values)
            aggregated_metrics[f'std_{key}'] = np.std(values)
        
        # Update best performance tracking
        mean_sharpe = aggregated_metrics.get('mean_sharpe_ratio', -np.inf)
        if mean_sharpe > self.best_performance['sharpe_ratio']:
            self.best_performance.update({
                'sharpe_ratio': mean_sharpe,
                'model_path': 'current_best',
                'parameters': self._get_model_parameters(),
                'timestamp': datetime.now()
            })
        
        return aggregated_metrics
    
    def _get_model_parameters(self) -> Dict:
        """Get current model parameters for tracking."""
        params = {
            'algorithm': self.algorithm,
            'learning_rate': self.model.learning_rate,
            'gamma': self.gamma
        }
        
        if hasattr(self.model, 'batch_size'):
            params['batch_size'] = self.model.batch_size
        
        return params
    
    def predict_trading_action(self, 
                             observation: np.ndarray, 
                             deterministic: bool = True) -> Tuple[int, float]:
        """Predict trading action for given observation."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        
        # Get action probabilities if available
        if hasattr(self.model.policy, 'predict_proba'):
            try:
                probs = self.model.policy.predict_proba(observation)
                confidence = np.max(probs)
            except:
                confidence = 0.5  # Default confidence
        else:
            confidence = 0.7  # Assume reasonable confidence for deterministic policies
        
        return int(action), confidence
    
    def save_trainer_state(self, filepath: str) -> None:
        """Save complete trainer state."""
        state = {
            'algorithm': self.algorithm,
            'parameters': self._get_model_parameters(),
            'training_history': self.training_history,
            'adaptation_history': self.adaptation_history,
            'best_performance': self.best_performance,
            'enhancement_enabled': self.enhancement_enabled
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Trainer state saved to {filepath}")
    
    def load_trainer_state(self, filepath: str) -> None:
        """Load trainer state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.training_history = state.get('training_history', [])
        self.adaptation_history = state.get('adaptation_history', [])
        self.best_performance = state.get('best_performance', self.best_performance)
        self.enhancement_enabled = state.get('enhancement_enabled', True)
        
        print(f"Trainer state loaded from {filepath}")
    
    def plot_training_progress(self, save_path: str = None) -> None:
        """Plot training progress and performance metrics."""
        if not self.training_history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract metrics
        steps = [h['step'] for h in self.training_history]
        sharpe_ratios = [h.get('sharpe_ratio', 0) for h in self.training_history]
        total_returns = [h.get('total_return', 0) for h in self.training_history]
        win_rates = [h.get('win_rate', 0) for h in self.training_history]
        noise_resilience = [h.get('noise_resilience_score', 0) for h in self.training_history]
        
        # Plot metrics
        axes[0, 0].plot(steps, sharpe_ratios)
        axes[0, 0].set_title('Sharpe Ratio Over Time')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        
        axes[0, 1].plot(steps, total_returns)
        axes[0, 1].set_title('Total Return Over Time')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Total Return')
        
        axes[1, 0].plot(steps, win_rates)
        axes[1, 0].set_title('Win Rate Over Time')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Win Rate')
        
        axes[1, 1].plot(steps, noise_resilience)
        axes[1, 1].set_title('Noise Resilience Score Over Time')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Noise Resilience Score')
        
        # Mark enhancement points
        for enhancement in self.adaptation_history:
            step = enhancement.get('step', 0)
            for ax in axes.flat:
                ax.axvline(x=step, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {save_path}")
        
        plt.show()

# Integration with trading guidance
class RLEnhancedGuidanceEngine(TradingGuidanceEngine):
    """Trading guidance engine enhanced with RL-optimized decision making."""
    
    def __init__(self, rl_trainer: AdaptiveRLTrainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl_trainer = rl_trainer
        self.rl_weight = 0.3  # Weight for RL vs rule-based decisions
    
    def get_enhanced_guidance(self, 
                            observation: np.ndarray,
                            pred_change: float, 
                            confidence: float, 
                            volatility: float) -> Tuple[str, str, Dict[str, Any]]:
        """Get guidance combining RL and rule-based approaches."""
        
        # Get rule-based guidance
        rule_action, rule_rationale, rule_metrics = self.get_guidance(
            pred_change, confidence, volatility
        )
        
        # Get RL-based guidance
        if self.rl_trainer.model is not None:
            rl_action_idx, rl_confidence = self.rl_trainer.predict_trading_action(observation)
            rl_actions = ['Hold', 'Buy', 'Sell']
            rl_action = rl_actions[rl_action_idx]
        else:
            rl_action = 'Hold'
            rl_confidence = 0.5
        
        # Combine decisions
        if rule_action == rl_action:
            # Both agree
            final_action = rule_action
            combined_confidence = (confidence + rl_confidence) / 2
            final_rationale = f"Consensus: {rule_rationale} (RL confirms with {rl_confidence:.1%} confidence)"
        else:
            # Disagreement - use weighted decision
            if rl_confidence > 0.8:
                final_action = rl_action
                final_rationale = f"RL override: {rl_action} (high RL confidence {rl_confidence:.1%}) vs rule-based {rule_action}"
                combined_confidence = rl_confidence
            else:
                final_action = rule_action
                final_rationale = f"Rule-based: {rule_rationale} (RL suggests {rl_action} with {rl_confidence:.1%} confidence)"
                combined_confidence = confidence
        
        enhanced_metrics = rule_metrics.copy()
        enhanced_metrics.update({
            'rl_action': rl_action,
            'rl_confidence': rl_confidence,
            'rule_action': rule_action,
            'combined_confidence': combined_confidence
        })
        
        return final_action, final_rationale, enhanced_metrics

# Usage example
if __name__ == '__main__':
    # Create sample training data
    from features.data_ingestion.pipeline import build_dataset
    
    print("Creating sample data for RL training...")
    data = build_dataset('AAPL')
    
    if not data.empty:
        # Initialize and train RL agent
        trainer = AdaptiveRLTrainer(algorithm='PPO', learning_rate=1e-4)
        
        print("Starting RL training...")
        results = trainer.train(
            data=data,
            total_timesteps=20000,
            eval_freq=2000,
            save_path="models/"
        )
        
        print("Training completed!")
        print(f"Final metrics: {results['final_metrics']}")
        
        # Plot training progress
        trainer.plot_training_progress()
        
        # Save trainer state
        trainer.save_trainer_state("trainer_state.pkl")
        
    else:
        print("No data available for training")