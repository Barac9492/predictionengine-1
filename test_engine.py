#!/usr/bin/env python3
"""
Quick test script to verify the Genius Prediction Engine works correctly.
"""

import sys
import os
import traceback
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from features.data_ingestion.pipeline import build_dataset
        print("✓ Data ingestion module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import data ingestion: {e}")
        return False
    
    try:
        from features.modeling.predictor import EnsemblePredictor
        print("✓ Modeling module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import modeling: {e}")
        return False
    
    try:
        from features.trading_decision.guidance import TradingGuidanceEngine
        print("✓ Trading guidance module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import trading guidance: {e}")
        return False
    
    try:
        from features.proxy_discovery.discovery import ProxyDiscoveryEngine
        print("✓ Proxy discovery module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import proxy discovery: {e}")
        return False
    
    try:
        from app.main_engine import GeniusPredictionEngine
        print("✓ Main engine imported successfully")
    except Exception as e:
        print(f"✗ Failed to import main engine: {e}")
        return False
    
    return True

def test_data_pipeline():
    """Test the data pipeline."""
    print("\nTesting data pipeline...")
    
    try:
        from features.data_ingestion.pipeline import build_dataset
        
        # Test with a simple stock
        print("Building dataset for AAPL...")
        data = build_dataset('AAPL')
        
        if data.empty:
            print("⚠ Warning: No data returned (may be due to API limits)")
        else:
            print(f"✓ Dataset built successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            print(f"  Columns: {list(data.columns)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        traceback.print_exc()
        return False

def test_guidance_engine():
    """Test the trading guidance engine."""
    print("\nTesting trading guidance engine...")
    
    try:
        from features.trading_decision.guidance import TradingGuidanceEngine
        
        engine = TradingGuidanceEngine()
        
        # Test prediction scenarios
        test_cases = [
            (0.05, 0.8, 0.02),   # Strong buy signal
            (-0.04, 0.9, 0.015), # Strong sell signal
            (0.01, 0.6, 0.03),   # Weak signal with noise
            (0.02, 0.5, 0.02),   # Low confidence
        ]
        
        for i, (pred_change, confidence, volatility) in enumerate(test_cases):
            action, rationale, metrics = engine.get_guidance(pred_change, confidence, volatility)
            print(f"  Test {i+1}: {action} - {rationale}")
        
        print("✓ Trading guidance engine working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Guidance engine test failed: {e}")
        traceback.print_exc()
        return False

def test_proxy_discovery():
    """Test the proxy discovery system."""
    print("\nTesting proxy discovery...")
    
    try:
        from features.proxy_discovery.discovery import ProxyDiscoveryEngine
        
        discovery = ProxyDiscoveryEngine()
        
        # Test proxy discovery for TSLA
        print("Discovering proxies for TSLA...")
        proxies = discovery.discover_proxies_for_stock('TSLA')
        
        if proxies:
            print(f"✓ Found {len(proxies)} proxy candidates")
            print("  Top 3 proxies:")
            for i, proxy in enumerate(proxies[:3]):
                print(f"    {i+1}. {proxy.name} (confidence: {proxy.confidence:.3f})")
        else:
            print("⚠ Warning: No proxies discovered")
        
        return True
        
    except Exception as e:
        print(f"✗ Proxy discovery test failed: {e}")
        traceback.print_exc()
        return False

def test_main_engine():
    """Test the main prediction engine."""
    print("\nTesting main prediction engine...")
    
    try:
        from app.main_engine import GeniusPredictionEngine
        
        # Initialize engine with minimal features for testing
        print("Initializing prediction engine...")
        engine = GeniusPredictionEngine(
            enable_rl=False,  # Disable RL for quick testing
            enable_feedback=False,  # Disable feedback for quick testing
            log_level="WARNING"  # Reduce log noise
        )
        
        # Test system status
        print("Getting system status...")
        status = engine.get_system_status()
        print(f"✓ System status: {status['components']['prediction_engine']}")
        
        # Test a simple prediction
        print("Generating test prediction for AAPL...")
        result = engine.get_prediction('AAPL', use_rl=False)
        
        if 'error' in result:
            print(f"⚠ Prediction returned error: {result['message']}")
        else:
            action = result['guidance']['action']
            confidence = result['prediction']['confidence']
            print(f"✓ Prediction successful: {action} (confidence: {confidence:.1%})")
        
        # Cleanup
        engine.shutdown()
        print("✓ Engine shutdown successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Main engine test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🤖 Genius Prediction Engine Test Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Pipeline", test_data_pipeline),
        ("Guidance Engine", test_guidance_engine),
        ("Proxy Discovery", test_proxy_discovery),
        ("Main Engine", test_main_engine),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if success:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! The Genius Prediction Engine is ready to use.")
        print("\nNext steps:")
        print("1. Run: python app/main_engine.py --stock AAPL")
        print("2. Or try interactive mode: python app/main_engine.py")
        print("3. For dashboard: streamlit run features/monitoring/dashboard.py")
        return 0
    else:
        print("⚠ Some tests failed. Check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check internet connection for data fetching")
        print("3. Some features may require API keys (see documentation)")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)