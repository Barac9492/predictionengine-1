# features/modeling/predictor.py
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from river import linear_model, metrics, optim
from shared.config.targets import VOL_ADJUST_THRESHOLD, CONFIDENCE_MIN

# Bayesian LSTM Model (using Pyro for probabilistic outputs)
class BayesianLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BayesianLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def bayesian_model(data, target=None):
    """Bayesian neural network model for probabilistic predictions."""
    input_size = data.shape[2] if len(data.shape) > 2 else data.shape[1]
    hidden_size = 32
    output_size = 1
    
    # Weight priors
    w1 = pyro.sample("w1", dist.Normal(0., 1.).expand([input_size, hidden_size]).to_event(2))
    b1 = pyro.sample("b1", dist.Normal(0., 1.).expand([hidden_size]).to_event(1))
    w2 = pyro.sample("w2", dist.Normal(0., 1.).expand([hidden_size, output_size]).to_event(2))
    b2 = pyro.sample("b2", dist.Normal(0., 1.).expand([output_size]).to_event(1))
    
    # Neural network forward pass
    if len(data.shape) > 2:
        data_flat = data.reshape(data.shape[0], -1)  # Flatten for fully connected layers
    else:
        data_flat = data
        
    hidden = torch.relu(torch.matmul(data_flat, w1) + b1)
    mu = torch.matmul(hidden, w2) + b2
    mu = mu.squeeze(-1)
    
    # Noise parameter
    sigma = pyro.sample("sigma", dist.HalfNormal(0.5))
    
    # Likelihood
    with pyro.plate("data", len(mu)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=target)
        
def bayesian_guide(data, target=None):
    """Variational guide for Bayesian neural network."""
    input_size = data.shape[2] if len(data.shape) > 2 else data.shape[1]
    hidden_size = 32
    output_size = 1
    
    # Variational parameters
    w1_mu = pyro.param("w1_mu", torch.randn(input_size, hidden_size))
    w1_sigma = pyro.param("w1_sigma", torch.ones(input_size, hidden_size), constraint=dist.constraints.positive)
    pyro.sample("w1", dist.Normal(w1_mu, w1_sigma).to_event(2))
    
    b1_mu = pyro.param("b1_mu", torch.randn(hidden_size))
    b1_sigma = pyro.param("b1_sigma", torch.ones(hidden_size), constraint=dist.constraints.positive)
    pyro.sample("b1", dist.Normal(b1_mu, b1_sigma).to_event(1))
    
    w2_mu = pyro.param("w2_mu", torch.randn(hidden_size, output_size))
    w2_sigma = pyro.param("w2_sigma", torch.ones(hidden_size, output_size), constraint=dist.constraints.positive)
    pyro.sample("w2", dist.Normal(w2_mu, w2_sigma).to_event(2))
    
    b2_mu = pyro.param("b2_mu", torch.randn(output_size))
    b2_sigma = pyro.param("b2_sigma", torch.ones(output_size), constraint=dist.constraints.positive)
    pyro.sample("b2", dist.Normal(b2_mu, b2_sigma).to_event(1))
    
    sigma_mu = pyro.param("sigma_mu", torch.tensor(0.1), constraint=dist.constraints.positive)
    pyro.sample("sigma", dist.HalfNormal(sigma_mu))

# Training function for Bayesian model
def train_bayesian_model(X, y, epochs=500, lr=0.01):
    """Train Bayesian neural network using SVI."""
    pyro.clear_param_store()
    
    # Convert to tensors
    X_tensor = torch.tensor(X).float()
    y_tensor = torch.tensor(y).float()
    
    # Setup SVI
    svi = SVI(bayesian_model, bayesian_guide, Adam({"lr": lr}), Trace_ELBO())
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        loss = svi.step(X_tensor, y_tensor)
        losses.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return svi, losses

# XGBoost Model for ensemble
def train_xgboost(X, y):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    return model

# Enhanced Online Predictor (from prototype, with probabilistic output)
class OnlinePredictor:
    def __init__(self):
        self.model = linear_model.LinearRegression(optimizer=optim.SGD(0.01))
        self.metric = metrics.MAE()
        self.conf_history = []

    def update_and_predict(self, features, target=None):
        if target is not None:
            pred = self.model.predict_one(features)
            self.model.learn_one(features, target)
            error = abs(target - pred)
            self.metric.update(target, pred)
            conf = max(0, 1 - error / 10)
            self.conf_history.append(conf)
        else:
            pred = self.model.predict_one(features)
            conf = np.mean(self.conf_history[-10:]) if self.conf_history else 0.5
            return pred, conf  # Mean, simulated std/conf

# Ensemble Predictor
class EnsemblePredictor:
    def __init__(self, features, target_col='returns'):
        self.scaler = MinMaxScaler()
        self.target_col = target_col
        self.bayesian_model = None
        self.xgboost_model = None
        self.online_model = OnlinePredictor()
        self.fit(features, features[target_col])  # Initial fit

    def prepare_data(self, df, seq_length=10):
        """Prepare data for modeling with proper feature scaling."""
        # Separate features and target
        feature_cols = [col for col in df.columns if col != self.target_col]
        target_col_idx = df.columns.get_loc(self.target_col)
        
        # Scale features only (not target)
        features_scaled = self.scaler.fit_transform(df[feature_cols])
        target_values = df[self.target_col].values
        
        # Create sequences for time series
        X, y = [], []
        for i in range(seq_length, len(features_scaled)):
            X.append(features_scaled[i-seq_length:i].flatten())  # Flatten the sequence
            y.append(target_values[i])
        
        return np.array(X), np.array(y)

    def fit(self, df, epochs=300):
        """Fit ensemble models on the dataset."""
        print(f"Fitting ensemble models on {len(df)} samples...")
        
        X, y = self.prepare_data(df)
        if len(X) == 0:
            print("Warning: No data available for training")
            return
            
        print(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # Train Bayesian model
        print("Training Bayesian model...")
        self.bayesian_model, self.losses = train_bayesian_model(X, y, epochs)
        
        # Train XGBoost
        print("Training XGBoost model...")
        self.xgboost_model = train_xgboost(X, y)
        
        # Fit online model
        print("Fitting online model...")
        for i in range(len(X)):
            features = {f'f{j}': X[i][j] for j in range(len(X[i]))}
            self.online_model.update_and_predict(features, y[i])
        
        print("Ensemble training completed.")

    def predict(self, new_data, n_samples=100):
        """Generate predictions with uncertainty estimates."""
        if self.bayesian_model is None or self.xgboost_model is None:
            print("Warning: Models not trained yet")
            return 0.0, 1.0
            
        # Prepare input - only use feature columns
        feature_cols = [col for col in new_data.columns if col != self.target_col]
        new_features = new_data[feature_cols]
        
        if len(new_features) < 10:  # Need enough data for sequence
            print("Warning: Not enough data for prediction")
            return 0.0, 1.0
            
        scaled = self.scaler.transform(new_features)
        input_seq = scaled[-10:].flatten().reshape(1, -1)  # Last 10 timesteps, flattened
        
        predictions = []
        
        # Bayesian prediction with uncertainty
        if self.bayesian_model is not None:
            try:
                # Sample from posterior
                predictive = pyro.infer.Predictive(bayesian_model, guide=bayesian_guide, num_samples=n_samples)
                input_tensor = torch.tensor(input_seq).float()
                samples = predictive(input_tensor)
                
                bayes_preds = samples['obs'].detach().numpy()
                bayes_mean = np.mean(bayes_preds)
                bayes_std = np.std(bayes_preds)
                predictions.append(('bayesian', bayes_mean, bayes_std))
            except Exception as e:
                print(f"Bayesian prediction failed: {e}")
                predictions.append(('bayesian', 0.0, 1.0))
        
        # XGBoost prediction
        if self.xgboost_model is not None:
            try:
                xgb_pred = self.xgboost_model.predict(input_seq)[0]
                predictions.append(('xgboost', xgb_pred, 0.1))  # Assume small uncertainty
            except Exception as e:
                print(f"XGBoost prediction failed: {e}")
                predictions.append(('xgboost', 0.0, 0.5))
        
        # Online prediction
        try:
            features = {f'f{j}': input_seq.flatten()[j] for j in range(len(input_seq.flatten()))}
            online_mean, online_conf = self.online_model.update_and_predict(features)
            online_std = 1 - online_conf  # Convert confidence to uncertainty
            predictions.append(('online', online_mean, online_std))
        except Exception as e:
            print(f"Online prediction failed: {e}")
            predictions.append(('online', 0.0, 0.5))
        
        # Ensemble prediction with uncertainty propagation
        if predictions:
            means = [pred[1] for pred in predictions]
            stds = [pred[2] for pred in predictions]
            
            # Weighted average (equal weights for now)
            ensemble_mean = np.mean(means)
            # Uncertainty propagation: sqrt of sum of variances
            ensemble_std = np.sqrt(np.mean([std**2 for std in stds]))
        else:
            ensemble_mean, ensemble_std = 0.0, 1.0
        
        return ensemble_mean, ensemble_std

# Usage example (integrate with pipeline)
if __name__ == '__main__':
    from features.data_ingestion.pipeline import build_dataset
    
    print("Building dataset...")
    df = build_dataset('AAPL')
    
    if not df.empty:
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Create predictor
        predictor = EnsemblePredictor(df, target_col='returns')
        
        # Make prediction
        pred_mean, pred_std = predictor.predict(df.iloc[-20:])  # Use last 20 rows
        print(f"Predicted return: {pred_mean:.4f} ± {pred_std:.4f}")
        
        # Convert to confidence score
        confidence = 1 / (1 + pred_std)
        print(f"Confidence: {confidence:.2%}")
    else:
        print("No data available for prediction")
