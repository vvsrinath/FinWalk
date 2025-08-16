#!/usr/bin/env python3
"""Test script to debug SMC strategy error"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import smart_money_concepts_strategy, fetch_stock_data
import traceback

def test_smc_strategy():
    """Test the SMC strategy to find the exact error"""
    try:
        print("Fetching data for AAPL...")
        data = fetch_stock_data("AAPL", "1d")
        print(f"Data shape: {data.shape}")
        
        print("Testing SMC strategy...")
        result = smart_money_concepts_strategy(data, "1d")
        print("SMC strategy result:", result)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_smc_strategy()