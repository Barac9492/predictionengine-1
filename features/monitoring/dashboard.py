# features/monitoring/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import sqlite3
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from features.data_ingestion.pipeline import build_dataset
from features.modeling.predictor import EnsemblePredictor
from features.trading_decision.guidance import TradingGuidanceEngine, AdvancedGuidanceEngine
from features.proxy_discovery.discovery import ProxyDiscoveryEngine

class PredictionEngineMonitor:
    """
    Comprehensive monitoring and feedback system for the prediction engine.
    Provides real-time tracking, performance analytics, and interactive dashboard.
    """
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.init_database()
        
        # Component instances
        self.guidance_engine = TradingGuidanceEngine()
        self.proxy_discovery = ProxyDiscoveryEngine()
        
        # Dashboard state
        self.current_stock = "AAPL"
        self.refresh_interval = 300  # 5 minutes
        
    def init_database(self):
        """Initialize SQLite database for tracking predictions and performance."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stock TEXT NOT NULL,
                predicted_change REAL,
                confidence REAL,
                volatility REAL,
                action TEXT,
                rationale TEXT,
                actual_change REAL,
                accuracy_score REAL,
                model_version TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stock TEXT NOT NULL,
                metric_name TEXT,
                metric_value REAL,
                period TEXT
            )
        ''')
        
        # Proxy performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proxy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stock TEXT NOT NULL,
                proxy_name TEXT,
                correlation_score REAL,
                signal_to_noise_ratio REAL,
                data_availability REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, 
                      stock: str,
                      predicted_change: float,
                      confidence: float,
                      volatility: float,
                      action: str,
                      rationale: str,
                      model_version: str = "v1.0"):
        """Log a new prediction to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (stock, predicted_change, confidence, volatility, action, rationale, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (stock, predicted_change, confidence, volatility, action, rationale, model_version))
        
        conn.commit()
        conn.close()
    
    def update_prediction_outcome(self, prediction_id: int, actual_change: float):
        """Update prediction with actual outcome."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate accuracy score
        cursor.execute('SELECT predicted_change, action FROM predictions WHERE id = ?', (prediction_id,))
        result = cursor.fetchone()
        
        if result:
            predicted_change, action = result
            
            # Calculate accuracy based on action correctness
            if action == "Buy" and actual_change > 0:
                accuracy_score = min(1.0, actual_change / max(abs(predicted_change), 0.01))
            elif action == "Sell" and actual_change < 0:
                accuracy_score = min(1.0, abs(actual_change) / max(abs(predicted_change), 0.01))
            elif action == "Hold":
                accuracy_score = 1.0 - abs(actual_change) * 10  # Reward for avoiding volatility
            else:
                accuracy_score = -abs(actual_change) * 5  # Penalty for wrong direction
            
            accuracy_score = max(0.0, min(1.0, accuracy_score))  # Clip to [0, 1]
            
            cursor.execute('''
                UPDATE predictions 
                SET actual_change = ?, accuracy_score = ?
                WHERE id = ?
            ''', (actual_change, accuracy_score, prediction_id))
        
        conn.commit()
        conn.close()
    
    def get_performance_metrics(self, stock: str, days: int = 30) -> Dict:
        """Get comprehensive performance metrics for a stock."""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent predictions
        cutoff_date = datetime.now() - timedelta(days=days)
        query = '''
            SELECT * FROM predictions 
            WHERE stock = ? AND timestamp > ? AND actual_change IS NOT NULL
            ORDER BY timestamp DESC
        '''
        
        predictions_df = pd.read_sql_query(query, conn, params=(stock, cutoff_date))
        conn.close()
        
        if predictions_df.empty:
            return {"error": "No data available"}
        
        # Calculate metrics
        metrics = {}
        
        # Overall accuracy
        metrics['overall_accuracy'] = predictions_df['accuracy_score'].mean()
        
        # Action-specific metrics
        for action in ['Buy', 'Sell', 'Hold']:
            action_data = predictions_df[predictions_df['action'] == action]
            if not action_data.empty:
                metrics[f'{action.lower()}_accuracy'] = action_data['accuracy_score'].mean()
                metrics[f'{action.lower()}_count'] = len(action_data)
        
        # Confidence vs accuracy correlation
        if len(predictions_df) > 5:
            conf_acc_corr = predictions_df['confidence'].corr(predictions_df['accuracy_score'])
            metrics['confidence_calibration'] = conf_acc_corr if not pd.isna(conf_acc_corr) else 0.0
        
        # Prediction vs actual correlation
        pred_actual_corr = predictions_df['predicted_change'].corr(predictions_df['actual_change'])
        metrics['prediction_correlation'] = pred_actual_corr if not pd.isna(pred_actual_corr) else 0.0
        
        # Time-based trends
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        predictions_df = predictions_df.sort_values('timestamp')
        
        if len(predictions_df) >= 10:
            # Recent vs earlier performance
            mid_point = len(predictions_df) // 2
            early_acc = predictions_df.iloc[:mid_point]['accuracy_score'].mean()
            recent_acc = predictions_df.iloc[mid_point:]['accuracy_score'].mean()
            metrics['performance_trend'] = recent_acc - early_acc
        
        # Volatility handling
        high_vol_data = predictions_df[predictions_df['volatility'] > predictions_df['volatility'].median()]
        if not high_vol_data.empty:
            metrics['high_volatility_accuracy'] = high_vol_data['accuracy_score'].mean()
        
        return metrics
    
    def create_dashboard(self):
        """Create Streamlit dashboard for monitoring."""
        st.set_page_config(
            page_title="Genius Prediction Engine Monitor",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("🤖 Genius Prediction Engine Monitor")
        st.markdown("Real-time monitoring and performance analytics")
        
        # Sidebar controls
        st.sidebar.header("Controls")
        
        # Stock selection
        available_stocks = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]
        selected_stock = st.sidebar.selectbox("Select Stock", available_stocks, index=0)
        
        # Time period
        time_period = st.sidebar.selectbox("Time Period", [7, 14, 30, 60, 90], index=2)
        
        # Refresh button
        if st.sidebar.button("Refresh Data"):
            st.rerun()
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto-refresh (5 min)", value=False)
        
        # Main dashboard
        col1, col2, col3 = st.columns(3)
        
        # Get performance metrics
        metrics = self.get_performance_metrics(selected_stock, time_period)
        
        # Key metrics cards
        with col1:
            if 'overall_accuracy' in metrics:
                st.metric("Overall Accuracy", f"{metrics['overall_accuracy']:.1%}")
            else:
                st.metric("Overall Accuracy", "No data")
        
        with col2:
            if 'prediction_correlation' in metrics:
                st.metric("Prediction Correlation", f"{metrics['prediction_correlation']:.3f}")
            else:
                st.metric("Prediction Correlation", "No data")
        
        with col3:
            if 'confidence_calibration' in metrics:
                st.metric("Confidence Calibration", f"{metrics['confidence_calibration']:.3f}")
            else:
                st.metric("Confidence Calibration", "No data")
        
        # Performance charts
        self._create_performance_charts(selected_stock, time_period)
        
        # Live prediction section
        st.header("Live Prediction")
        self._create_live_prediction_section(selected_stock)
        
        # Proxy performance
        st.header("Proxy Performance")
        self._create_proxy_performance_section(selected_stock)
        
        # System health
        st.header("System Health")
        self._create_system_health_section()
        
        # Auto-refresh logic
        if auto_refresh:
            import time
            time.sleep(self.refresh_interval)
            st.rerun()
    
    def _create_performance_charts(self, stock: str, days: int):
        """Create performance visualization charts."""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        query = '''
            SELECT * FROM predictions 
            WHERE stock = ? AND timestamp > ? AND actual_change IS NOT NULL
            ORDER BY timestamp
        '''
        
        df = pd.read_sql_query(query, conn, params=(stock, cutoff_date))
        conn.close()
        
        if df.empty:
            st.warning("No prediction data available for the selected period")
            return
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy over time
            fig_acc = go.Figure()
            
            # Rolling accuracy
            df['rolling_accuracy'] = df['accuracy_score'].rolling(window=5, min_periods=1).mean()
            
            fig_acc.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['rolling_accuracy'],
                mode='lines+markers',
                name='Rolling Accuracy (5-day)',
                line=dict(color='blue')
            ))
            
            fig_acc.update_layout(
                title="Prediction Accuracy Over Time",
                xaxis_title="Date",
                yaxis_title="Accuracy Score",
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig_acc, use_container_width=True)
        
        with col2:
            # Prediction vs Actual scatter
            fig_scatter = go.Figure()
            
            colors = {'Buy': 'green', 'Sell': 'red', 'Hold': 'blue'}
            
            for action in df['action'].unique():
                action_data = df[df['action'] == action]
                fig_scatter.add_trace(go.Scatter(
                    x=action_data['predicted_change'],
                    y=action_data['actual_change'],
                    mode='markers',
                    name=action,
                    marker=dict(color=colors.get(action, 'gray'), size=8),
                    text=action_data['timestamp'].dt.strftime('%Y-%m-%d'),
                    hovertemplate='<b>%{text}</b><br>Predicted: %{x:.3f}<br>Actual: %{y:.3f}<extra></extra>'
                ))
            
            # Add diagonal line for perfect prediction
            min_val = min(df['predicted_change'].min(), df['actual_change'].min())
            max_val = max(df['predicted_change'].max(), df['actual_change'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='gray')
            ))
            
            fig_scatter.update_layout(
                title="Predicted vs Actual Changes",
                xaxis_title="Predicted Change",
                yaxis_title="Actual Change"
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Action distribution
        action_counts = df['action'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=action_counts.index,
            values=action_counts.values,
            textinfo='label+percent'
        )])
        fig_pie.update_layout(title="Action Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    def _create_live_prediction_section(self, stock: str):
        """Create live prediction interface."""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Generate New Prediction", type="primary"):
                with st.spinner("Generating prediction..."):
                    try:
                        # Build dataset
                        data = build_dataset(stock)
                        
                        if not data.empty:
                            # Create predictor
                            predictor = EnsemblePredictor(data, target_col='returns')
                            
                            # Make prediction
                            pred_mean, pred_std = predictor.predict(data.iloc[-20:])
                            confidence = 1 / (1 + pred_std)
                            volatility = data['volatility'].iloc[-1]
                            
                            # Get guidance
                            action, rationale, metrics = self.guidance_engine.get_guidance(
                                pred_mean, confidence, volatility
                            )
                            
                            # Display results
                            st.success("Prediction Generated!")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Predicted Change", f"{pred_mean:.2%}")
                            with col_b:
                                st.metric("Confidence", f"{confidence:.1%}")
                            with col_c:
                                st.metric("Recommended Action", action)
                            
                            st.info(f"**Rationale:** {rationale}")
                            
                            # Log prediction
                            self.log_prediction(
                                stock, pred_mean, confidence, volatility,
                                action, rationale
                            )
                            
                        else:
                            st.error("Unable to fetch data for prediction")
                            
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
        
        with col2:
            st.markdown("### Latest Predictions")
            
            # Show recent predictions
            conn = sqlite3.connect(self.db_path)
            recent_query = '''
                SELECT timestamp, action, predicted_change, confidence 
                FROM predictions 
                WHERE stock = ? 
                ORDER BY timestamp DESC 
                LIMIT 5
            '''
            recent_df = pd.read_sql_query(recent_query, conn, params=(stock,))
            conn.close()
            
            if not recent_df.empty:
                recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%H:%M')
                st.dataframe(recent_df, hide_index=True)
            else:
                st.info("No recent predictions")
    
    def _create_proxy_performance_section(self, stock: str):
        """Create proxy performance monitoring."""
        try:
            # Discover proxies for the stock
            proxies = self.proxy_discovery.get_top_proxies_for_stock(stock, limit=5)
            
            if proxies:
                proxy_df = pd.DataFrame(proxies)
                
                # Display proxy rankings
                st.subheader("Top Proxies")
                
                for i, proxy in enumerate(proxies):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{proxy['name']}**")
                        st.caption(proxy['description'])
                    
                    with col2:
                        st.metric("Confidence", f"{proxy['confidence']:.1%}")
                    
                    with col3:
                        st.metric("Source", proxy['source'])
            else:
                st.info("No proxy data available")
                
        except Exception as e:
            st.warning(f"Proxy discovery unavailable: {str(e)}")
    
    def _create_system_health_section(self):
        """Create system health monitoring."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Database size
            try:
                db_size = os.path.getsize(self.db_path) / (1024 * 1024)  # MB
                st.metric("Database Size", f"{db_size:.1f} MB")
            except:
                st.metric("Database Size", "Unknown")
        
        with col2:
            # Recent activity
            conn = sqlite3.connect(self.db_path)
            recent_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM predictions WHERE timestamp > datetime('now', '-24 hours')",
                conn
            ).iloc[0]['count']
            conn.close()
            st.metric("24h Predictions", recent_count)
        
        with col3:
            # System status
            st.metric("System Status", "🟢 Online")

# Standalone dashboard runner
def run_dashboard():
    """Run the monitoring dashboard."""
    monitor = PredictionEngineMonitor()
    monitor.create_dashboard()

if __name__ == "__main__":
    run_dashboard()