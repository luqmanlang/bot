import os
import time
from coinmarketcap import Market
import pandas as pd
import ta  # Technical Analysis library

# --- Configuration ---
# Get your CoinMarketCap API key from environment variables for security
CMC_PRO_API_KEY = os.environ.get("CMC_PRO_API_KEY", "YOUR_CMC_PRO_API_KEY_HERE")
if "YOUR_CMC_PRO_API_KEY_HERE" in CMC_PRO_API_KEY:
    print("WARNING: Please set your CMC_PRO_API_KEY environment variable or replace the placeholder.")
    print("You can get your API key from CoinMarketCap Developer Portal: https://coinmarketcap.com/api/pricing/")

# --- Initialize CMC API Client ---
# Use the 'PRO' API for more features and historical data
coinmarketcap_client = Market(CMC_PRO_API_KEY)

# --- Functions for Data Retrieval ---
def get_crypto_data(symbol="BTC", convert="USD"):
    try:
        data = coinmarketcap_client.cryptocurrency_info(symbol=symbol)
        if data and data['data'] and symbol in data['data']:
            info = data['data'][symbol]

            # Get latest quotes for price, volume, market cap
            quotes = coinmarketcap_client.cryptocurrency_quotes_latest(symbol=symbol, convert=convert)
            if quotes and quotes['data'] and symbol in quotes['data']:
                quote_data = quotes['data'][symbol]['quote'][convert]
                return {
                    "name": info.get("name"),
                    "symbol": info.get("symbol"),
                    "price": quote_data.get("price", 0.0),
                    "percent_change_24h": quote_data.get("percent_change_24h", 0.0),
                    "volume_24h": quote_data.get("volume_24h", 0.0),
                    "market_cap": quote_data.get("market_cap", 0.0),
                }
        return None
    except Exception as e:
        print(f"Error fetching crypto data for {symbol}: {e}")
        return None

def get_historical_data(symbol="BTC", time_period="24h", interval="5m"):
    """
    Fetches historical data. Note: CoinMarketCap free API tier has limited historical data access.
    You might need a paid plan or use other sources like ccxt for more extensive historical data.
    """
    try:
        # CMC Historical data endpoint requires professional plan
        # For free tier, you might only get limited chart data or rely on other APIs.
        # This part is a placeholder as direct historical data is often a paid feature on CMC.
        print(f"Fetching historical data for {symbol} for {time_period} period with {interval} interval.")
        print("Note: CoinMarketCap free API has limitations on historical data.")
        # Example using placeholder data for demonstration
        data = {
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        }
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

# --- Functions for Technical Indicators ---
def calculate_technical_indicators(df):
    if df.empty or len(df) < 20: # Need enough data for indicators
        return {}

    indicators = {}
    # RSI(5,3,3) - The image shows RSI(5,3,3) which is unusual. RSI typically takes a single period.
    # Assuming it means RSI(5) for period, and then 3,3 are for smoothing or other parameters not standard in basic RSI.
    # We will use RSI(5) as a common interpretation. If you need specific RSI parameters like 5,3,3,
    # you might need to implement a custom RSI calculation or find a library that supports it.
    indicators["RSI"] = ta.momentum.RSIIndicator(close=df['close'], window=5).rsi().iloc[-1]

    # Stochastic(5)
    indicators["Stochastic_K"] = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=5, smooth_window=3
    ).stoch().iloc[-1]
    indicators["Stochastic_D"] = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=5, smooth_window=3
    ).stoch_signal().iloc[-1]


    # MACD (standard parameters 12, 26, 9)
    # The image shows MACD: 981.6416, which suggests it's the MACD line value, not the signal.
    # If you need specific MACD parameters, adjust them here.
    indicators["MACD"] = ta.trend.MACD(
        close=df['close'], window_fast=12, window_slow=26, window_sign=9
    ).macd().iloc[-1]
    indicators["MACD_Signal"] = ta.trend.MACD(
        close=df['close'], window_fast=12, window_slow=26, window_sign=9
    ).macd_signal().iloc[-1]

    return indicators

# --- Functions for Trading Signals ---
def generate_trading_signals(indicators):
    signal = "HOLD"
    confidence = 0
    reasons = []

    rsi = indicators.get("RSI")
    stoch_k = indicators.get("Stochastic_K")
    stoch_d = indicators.get("Stochastic_D")
    macd = indicators.get("MACD")
    macd_signal = indicators.get("MACD_Signal")

    # RSI Signal
    if rsi is not None:
        if rsi < 30:
            reasons.append("RSI (Oversold)")
            confidence += 15
        elif rsi > 70:
            reasons.append("RSI (Overbought)")
            confidence += 15
        else:
            reasons.append("RSI (Neutral)")

    # Stochastic Signal
    if stoch_k is not None and stoch_d is not None:
        if stoch_k < 20 and stoch_k > stoch_d:
            reasons.append("Stochastic (Bullish Crossover)")
            confidence += 20
        elif stoch_k > 80 and stoch_k < stoch_d:
            reasons.append("Stochastic (Bearish Crossover)")
            confidence += 20
        elif stoch_k > 80:
            reasons.append("Stochastic (Overbought)")
            signal = "Sell" # Stronger signal, temporary override
            confidence += 10
        elif stoch_k < 20:
            reasons.append("Stochastic (Oversold)")
            signal = "Buy" # Stronger signal, temporary override
            confidence += 10
        else:
            reasons.append("Stochastic (Neutral)")

    # MACD Signal
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            reasons.append("MACD (Bullish Crossover)")
            confidence += 25
            if signal != "Sell": # Don't override a strong sell signal with a weaker buy
                signal = "BUY"
        elif macd < macd_signal:
            reasons.append("MACD (Bearish Crossover)")
            confidence += 25
            if signal != "Buy": # Don't override a strong buy signal with a weaker sell
                signal = "SELL"
        else:
            reasons.append("MACD (Neutral)")

    # Combine signals and determine final recommendation
    if "Stochastic (Bearish Crossover)" in reasons and "MACD (Bearish Crossover)" in reasons:
        signal = "SELL"
    elif "Stochastic (Bullish Crossover)" in reasons and "MACD (Bullish Crossover)" in reasons:
        signal = "BUY"
    elif "Stochastic (Overbought)" in reasons and "MACD (Bearish Crossover)" in reasons:
        signal = "SELL" # Stronger inclination to sell
    elif "Stochastic (Oversold)" in reasons and "MACD (Bullish Crossover)" in reasons:
        signal = "BUY" # Stronger inclination to buy


    confidence = min(confidence, 100) # Cap confidence at 100%

    recommendation_text = f"Disyorkan untuk {signal} dengan keyakinan {confidence}%."
    if confidence < 60 and signal != "HOLD":
        recommendation_text += " Tunggu signal yang lebih jelas."
        signal = "HOLD" # If confidence is low, revert to HOLD

    return signal, confidence, recommendation_text, reasons

# --- Market Analysis (Simplified) ---
def perform_market_analysis(price_change_24h):
    sentiment = "Neutral"
    score = 0.0
    momentum = "Neutral"

    if price_change_24h > 5:
        sentiment = "Bullish"
        score = 2.0
        momentum = "Strong Up"
    elif price_change_24h > 0:
        sentiment = "Slightly Bullish"
        score = 1.0
        momentum = "Up"
    elif price_change_24h < -5:
        sentiment = "Bearish"
        score = -2.0
        momentum = "Strong Down"
    elif price_change_24h < 0:
        sentiment = "Slightly Bearish"
        score = -1.0
        momentum = "Down"

    return sentiment, score, momentum

# --- Risk Assessment (Simplified) ---
def perform_risk_assessment(historical_df):
    risk_level = "Medium"
    volatility = 0.0
    risk_score = 5

    if not historical_df.empty and len(historical_df) > 1:
        # Calculate volatility using standard deviation of log returns
        historical_df['log_returns'] = ta.others.log_return(close=historical_df['close'])
        volatility = historical_df['log_returns'].std() * (252**0.5) * 100 # Annualized percentage volatility

        if volatility < 1.0:
            risk_level = "Very Low"
            risk_score = 1
        elif volatility < 2.0:
            risk_level = "Low"
            risk_score = 2
        elif volatility < 5.0:
            risk_level = "Medium"
            risk_score = 4
        elif volatility < 10.0:
            risk_level = "High"
            risk_score = 7
        else:
            risk_level = "Very High"
            risk_score = 9

    return risk_level, volatility, risk_score

# --- Main Bot Logic ---
def run_crypto_analysis_bot(symbol="BTC"):
    print(f"--- Analisis Teknikal {symbol} (Manual) ---")

    # 1. Get Current Price Data
    current_data = get_crypto_data(symbol)
    if not current_data:
        print(f"Could not retrieve current data for {symbol}. Exiting.")
        return

    print(f"\n💰 Harga Semasa")
    print(f"${current_data['price']:.6f}")
    print(f"{current_data['percent_change_24h']:.2f}% (24j)")

    print(f"\n📊 Volume & Market Cap")
    print(f"Volume: ${current_data['volume_24h']:.2f}")
    print(f"Market Cap: ${current_data['market_cap']:.2f}")

    # 2. Get Historical Data for Technical Indicators and Risk Assessment
    # Note: Replace this with actual historical data fetching for production.
    historical_df = get_historical_data(symbol)

    print(f"\n📈 Petunjuk Teknikal")
    if not historical_df.empty:
        indicators = calculate_technical_indicators(historical_df)
        print(f"📊 RSI(5): {indicators.get('RSI', 'N/A'):.1f} ({'Neutral-Bearish' if indicators.get('RSI', 50) > 50 else 'Neutral-Bullish'})") # Simplified status
        print(f"📉 Stochastic(5): {indicators.get('Stochastic_K', 'N/A'):.1f} ({'Sell' if indicators.get('Stochastic_K', 50) > 80 else ('Buy' if indicators.get('Stochastic_K', 50) < 20 else 'Neutral')})")
        print(f"📈 MACD: {indicators.get('MACD', 'N/A'):.4f} ({'Strong Bullish' if indicators.get('MACD', 0) > indicators.get('MACD_Signal', 0) else 'Strong Bearish'})")
    else:
        indicators = {}
        print("Not enough historical data to calculate technical indicators.")

    # 3. Trading Signals
    print(f"\n➡️ Trading Signals")
    signal, confidence, recommendation_text, reasons = generate_trading_signals(indicators)
    print(f"💡 Signal: {signal}")
    print(f"🔑 Confidence: {confidence}%")
    print(f"📝 Recommendation: {recommendation_text}")

    # 4. Market Analysis
    print(f"\n🔥 Market Analysis")
    sentiment, score, momentum = perform_market_analysis(current_data['percent_change_24h'])
    print(f"💡 Market Sentiment: {sentiment}")
    print(f"📊 Score: {score:+.1f}")
    print(f"📈 Momentum: {momentum}")

    # 5. Risk Assessment
    print(f"\n✅ Risk Assessment")
    risk_level, volatility, risk_score = perform_risk_assessment(historical_df)
    print(f"🛡️ Risk Level: {risk_level}")
    print(f"⚠️ Volatility: {volatility:.1f}%")
    print(f"🔥 Risk Score: {risk_score}/10")

# --- Run the Bot ---
if __name__ == "__main__":
    # Ensure you have your CMC_PRO_API_KEY set as an environment variable
    # e.g., export CMC_PRO_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
    # Or replace "YOUR_CMC_PRO_API_KEY_HERE" directly in the code (less secure for production)
    run_crypto_analysis_bot("BTC")
