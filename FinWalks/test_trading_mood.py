#!/usr/bin/env python3
"""Test script for trading mood calculation"""

def test_trading_mood():
    """Test the trading mood calculation directly"""
    
    # Mock the calculate_trading_mood function for testing
    def calculate_trading_mood(signal, confidence, indicators, market_conditions):
        try:
            # Base mood calculation from signal and confidence
            mood_score = 0
            
            # Signal contribution (40% weight)
            if signal == "BUY":
                mood_score += 0.4 * confidence
            elif signal == "SELL":
                mood_score -= 0.4 * confidence
            
            # RSI contribution (20% weight)
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi > 70:
                    mood_score -= 0.2 * ((rsi - 70) / 30)  # Overbought = negative
                elif rsi < 30:
                    mood_score += 0.2 * ((30 - rsi) / 30)  # Oversold = positive
            
            # MACD contribution (20% weight)
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd_diff = indicators['macd'] - indicators['macd_signal']
                if macd_diff > 0:
                    mood_score += 0.2 * min(abs(macd_diff), 1)
                else:
                    mood_score -= 0.2 * min(abs(macd_diff), 1)
            
            # Normalize mood score to -1 to 1 range
            mood_score = max(-1, min(1, mood_score))
            
            # Determine emoji and description based on mood score
            if mood_score >= 0.8:
                return {
                    'emoji': '🚀',
                    'mood': 'Extremely Bullish',
                    'description': 'Strong upward momentum with high confidence',
                    'color': '#00ff00',
                    'score': mood_score
                }
            elif mood_score >= 0.6:
                return {
                    'emoji': '😄',
                    'mood': 'Very Bullish',
                    'description': 'Positive signals with good momentum',
                    'color': '#32cd32',
                    'score': mood_score
                }
            elif mood_score >= 0.4:
                return {
                    'emoji': '😊',
                    'mood': 'Bullish',
                    'description': 'Moderate bullish signals detected',
                    'color': '#90ee90',
                    'score': mood_score
                }
            elif mood_score >= 0.2:
                return {
                    'emoji': '🙂',
                    'mood': 'Slightly Bullish',
                    'description': 'Weak bullish tendency',
                    'color': '#98fb98',
                    'score': mood_score
                }
            elif mood_score >= -0.2:
                return {
                    'emoji': '😐',
                    'mood': 'Neutral',
                    'description': 'Mixed signals, market indecision',
                    'color': '#ffd700',
                    'score': mood_score
                }
            else:
                return {
                    'emoji': '😟',
                    'mood': 'Bearish',
                    'description': 'Moderate bearish signals detected',
                    'color': '#ff6347',
                    'score': mood_score
                }
        except Exception as e:
            print(f"Error: {e}")
            return {'emoji': '🤔', 'mood': 'Error', 'score': 0}
    
    # Test case 1: BUY signal with high confidence
    print("=== Test 1: BUY Signal with High Confidence ===")
    mood = calculate_trading_mood(
        signal="BUY",
        confidence=0.95,
        indicators={
            "rsi": 71.73,
            "macd": 2.6755,
            "macd_signal": 1.6662,
            "atr": 4.17,
            "sma_20": 204.32,
            "sma_50": 203.9
        },
        market_conditions={'trend': 'bullish', 'volatility': 'low'}
    )
    print(f"Result: {mood}")
    print()
    
    # Test case 2: SELL signal with medium confidence
    print("=== Test 2: SELL Signal with Medium Confidence ===")
    mood = calculate_trading_mood(
        signal="SELL",
        confidence=0.6,
        indicators={
            "rsi": 25,
            "macd": -1.5,
            "macd_signal": -0.5,
            "atr": 3.0,
            "sma_20": 200,
            "sma_50": 205
        },
        market_conditions={'trend': 'bearish', 'volatility': 'high'}
    )
    print(f"Result: {mood}")
    print()
    
    # Test case 3: Neutral signal
    print("=== Test 3: Neutral Signal ===")
    mood = calculate_trading_mood(
        signal="HOLD",
        confidence=0.3,
        indicators={
            "rsi": 50,
            "macd": 0.1,
            "macd_signal": 0.0,
            "atr": 2.0,
            "sma_20": 200,
            "sma_50": 200
        },
        market_conditions={'trend': 'neutral', 'volatility': 'normal'}
    )
    print(f"Result: {mood}")

if __name__ == "__main__":
    test_trading_mood()