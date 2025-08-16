#!/usr/bin/env python3
"""Test script for square root analysis functionality"""

import math

def test_square_root_levels():
    """Test the square root levels calculation"""
    current_price = 211.16
    recent_high = 225.0
    recent_low = 195.0
    
    print(f"Testing square root levels for price: ${current_price}")
    print(f"Recent high: ${recent_high}, Recent low: ${recent_low}")
    
    # Base square root of current price
    current_sqrt = math.sqrt(current_price)
    print(f"Current square root: {current_sqrt}")
    
    # Generate some square root levels
    sqrt_levels = []
    current_sqrt_floor = int(math.sqrt(current_price))
    
    for i in range(current_sqrt_floor - 5, current_sqrt_floor + 10):
        perfect_square = i ** 2
        sqrt_levels.append(perfect_square)
        print(f"Level {i}^2 = {perfect_square}")
    
    # Find support and resistance
    support_levels = [level for level in sqrt_levels if level < current_price]
    resistance_levels = [level for level in sqrt_levels if level > current_price]
    
    print(f"\nSupport levels: {support_levels[-3:]}")
    print(f"Resistance levels: {resistance_levels[:3]}")
    
    return {
        'current_sqrt': current_sqrt,
        'support_levels': support_levels[-3:],
        'resistance_levels': resistance_levels[:3],
        'total_levels': len(sqrt_levels)
    }

if __name__ == "__main__":
    result = test_square_root_levels()
    print(f"\nTest result: {result}")