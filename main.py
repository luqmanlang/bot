import os
import time
import pandas as pd
import ta

# --- IMPORT LIBRARY coinmarketcap ---
# Pastikan pustaka 'python-coinmarketcap' terinstal.
# Jika tidak terinstal, script akan memberikan pesan error yang jelas dan keluar.
try:
    from coinmarketcap import Market
except ImportError:
    print("------------------------------------------------------------------")
    print("ERROR: Pustaka 'python-coinmarketcap' tidak ditemukan.")
    print("Pastikan Anda sudah menginstal dengan 'pip install python-coinmarketcap'")
    print("dan file 'requirements.txt' Anda berisi 'python-coinmarketcap'.")
    print("------------------------------------------------------------------")
    exit(1) # Keluar dari program jika pustaka inti tidak ada

# --- KONFIGURASI API KEY ---
# API Key CoinMarketCap diambil dari environment variables untuk keamanan.
# JANGAN hardcode API Key Anda langsung di sini jika akan di-push ke Git publik.
CMC_PRO_API_KEY = os.environ.get("CMC_PRO_API_KEY")

if not CMC_PRO_API_KEY:
    print("------------------------------------------------------------------")
    print("ERROR: Variabel lingkungan 'CMC_PRO_API_KEY' tidak diatur.")
    print("Silakan atur API Key CoinMarketCap Anda sebagai environment variable.")
    print("Contoh: export CMC_PRO_API_KEY='INI_API_KEY_ANDA'")
    print("Atau di dashboard Render, tambahkan environment variable tersebut.")
    print("------------------------------------------------------------------")
    exit(1) # Keluar jika API Key tidak ditemukan

# --- INISIALISASI KLIEN CMC API ---
try:
    coinmarketcap_client = Market(CMC_PRO_API_KEY)
except Exception as e:
    print("------------------------------------------------------------------")
    print(f"ERROR: Gagal menginisialisasi klien CoinMarketCap: {e}")
    print("Periksa kembali API Key Anda dan koneksi jaringan.")
    print("------------------------------------------------------------------")
    exit(1)

# --- KONFIGURASI BOT ---
# Simbol kripto utama yang akan dianalisis (default: BTC)
TARGET_CRYPTO_SYMBOL = os.environ.get("TARGET_CRYPTO_SYMBOL", "BTC").upper()


# --- FUNGSI PENGAMBILAN DATA ---
def get_crypto_data(symbol, convert="USD"):
    """Mengambil data harga, volume, dan kapitalisasi pasar terkini untuk simbol tertentu."""
    try:
        quotes_data = coinmarketcap_client.cryptocurrency_quotes_latest(symbol=symbol, convert=convert)
        if not quotes_data or not quotes_data.get('data') or symbol not in quotes_data['data']:
            print(f"Tidak ada data quote untuk {symbol}.")
            return None

        quote_details = quotes_data['data'][symbol]['quote'][convert]
        name = quotes_data['data'][symbol].get('name', symbol)

        return {
            "name": name,
            "symbol": symbol,
            "price": quote_details.get("price", 0.0),
            "percent_change_24h": quote_details.get("percent_change_24h", 0.0),
            "volume_24h": quote_details.get("volume_24h", 0.0),
            "market_cap": quote_details.get("market_cap", 0.0),
        }
    except Exception as e:
        print(f"ERROR: Gagal mengambil data kripto terkini untuk {symbol}: {e}")
        return None

def get_historical_data(symbol, limit=60):
    """
    Mengambil data historis. Menggunakan data placeholder untuk demo/free tier.
    Untuk produksi, ganti dengan API riil (misal, CMC berbayar atau ccxt).
    """
    # print(f"Mencoba mendapatkan data historis untuk {symbol}. (Catatan: Batasan tier gratis CMC)")

    # Data Placeholder (minimal 60 poin untuk TA yang lebih baik)
    num_points = limit # Jumlah data poin untuk placeholder
    data = {
        'open': [100 + i for i in range(num_points)],
        'high': [102 + i for i in range(num_points)],
        'low': [99 + i for i in range(num_points)],
        'close': [101 + i + (i % 5) * 0.5 - (i % 7) * 0.2 for i in range(num_points)], # Variasi harga
        'volume': [1000 + i * 10 for i in range(num_points)]
    }
    df = pd.DataFrame(data)
    # Gunakan waktu saat ini untuk titik data terakhir
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    df['timestamp'] = pd.to_datetime(pd.date_range(end=end_time_str, periods=len(df), freq='5min'))
    df = df.set_index('timestamp').sort_index()
    # print(f"Menggunakan {len(df)} titik data placeholder historis untuk {symbol}.")
    return df

# --- FUNGSI INDIKATOR TEKNIKAL ---
def calculate_technical_indicators(df):
    """Menghitung indikator teknikal dari DataFrame historis."""
    if df.empty or len(df) < 20: # Perlu setidaknya 20 data poin untuk indikator yang masuk akal
        # print("Peringatan: Data tidak cukup untuk perhitungan indikator teknikal yang akurat.")
        return {}

    indicators = {}
    try:
        # RSI(5)
        indicators["RSI"] = ta.momentum.RSIIndicator(close=df['close'], window=5).rsi().iloc[-1]
    except Exception as e:
        print(f"Error menghitung RSI: {e}")
        indicators["RSI"] = None

    try:
        # Stochastic Oscillator (5,3,3)
        stoch_oscillator = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'], window=5, smooth_window=3
        )
        indicators["Stochastic_K"] = stoch_oscillator.stoch().iloc[-1]
        indicators["Stochastic_D"] = stoch_oscillator.stoch_signal().iloc[-1]
    except Exception as e:
        print(f"Error menghitung Stochastic: {e}")
        indicators["Stochastic_K"] = None
        indicators["Stochastic_D"] = None

    try:
        # MACD (12, 26, 9)
        macd_indicator = ta.trend.MACD(
            close=df['close'], window_fast=12, window_slow=26, window_sign=9
        )
        indicators["MACD"] = macd_indicator.macd().iloc[-1]
        indicators["MACD_Signal"] = macd_indicator.macd_signal().iloc[-1]
    except Exception as e:
        print(f"Error menghitung MACD: {e}")
        indicators["MACD"] = None
        indicators["MACD_Signal"] = None

    return indicators

# --- FUNGSI Sinyal Trading ---
def generate_trading_signals(indicators):
    """Menghasilkan sinyal trading berdasarkan indikator teknikal."""
    signal = "HOLD"
    confidence = 0
    indicator_statuses = {
        "RSI": "Neutral",
        "Stochastic": "Neutral",
        "MACD": "Neutral"
    }

    rsi = indicators.get("RSI")
    stoch_k = indicators.get("Stochastic_K")
    stoch_d = indicators.get("Stochastic_D")
    macd = indicators.get("MACD")
    macd_signal = indicators.get("MACD_Signal")

    # Analisis RSI
    if rsi is not None:
        if rsi < 30:
            indicator_statuses["RSI"] = "Oversold (Potensi Beli)"
            confidence += 20
        elif rsi > 70:
            indicator_statuses["RSI"] = "Overbought (Potensi Jual)"
            confidence += 20

    # Analisis Stochastic
    if stoch_k is not None and stoch_d is not None:
        if stoch_k < 20 and stoch_k > stoch_d: # Crossover bullish di zona oversold
            indicator_statuses["Stochastic"] = "Bullish Crossover (Beli Kuat)"
            confidence += 35
        elif stoch_k > 80 and stoch_k < stoch_d: # Crossover bearish di zona overbought
            indicator_statuses["Stochastic"] = "Bearish Crossover (Jual Kuat)"
            confidence += 35
        elif stoch_k > 80:
            indicator_statuses["Stochastic"] = "Overbought"
            confidence += 10
        elif stoch_k < 20:
            indicator_statuses["Stochastic"] = "Oversold"
            confidence += 10

    # Analisis MACD
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            indicator_statuses["MACD"] = "Bullish Crossover"
            confidence += 25
        elif macd < macd_signal:
            indicator_statuses["MACD"] = "Bearish Crossover"
            confidence += 25

    # Menentukan sinyal akhir berdasarkan konsensus
    buy_votes = 0
    sell_votes = 0

    if "Oversold" in indicator_statuses["RSI"]: buy_votes += 1
    if "Overbought" in indicator_statuses["RSI"]: sell_votes += 1

    if "Bullish Crossover" in indicator_statuses["Stochastic"] or "Oversold" in indicator_statuses["Stochastic"]: buy_votes += 1
    if "Bearish Crossover" in indicator_statuses["Stochastic"] or "Overbought" in indicator_statuses["Stochastic"]: sell_votes += 1

    if "Bullish Crossover" in indicator_statuses["MACD"]: buy_votes += 1
    if "Bearish Crossover" in indicator_statuses["MACD"]: sell_votes += 1

    # Logic for final signal
    if buy_votes >= 2: # Setidaknya 2 sinyal beli
        signal = "BUY"
    elif sell_votes >= 2: # Setidaknya 2 sinyal jual
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = min(confidence, 100) # Batasi kepercayaan hingga 100%

    recommendation_text = f"Disyorkan untuk {signal} dengan keyakinan {confidence}%."
    if confidence < 60 and signal != "HOLD":
        recommendation_text += " Tunggu signal yang lebih jelas."
        signal = "HOLD" # Jika kepercayaan rendah, kembali ke HOLD untuk keamanan

    return signal, confidence, recommendation_text, indicator_statuses

# --- ANALISIS PASAR ---
def perform_market_analysis(price_change_24h):
    """Melakukan analisis sentimen pasar sederhana berdasarkan perubahan harga 24 jam."""
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

# --- PENILAIAN RISIKO ---
def perform_risk_assessment(historical_df):
    """Melakukan penilaian risiko sederhana berdasarkan volatilitas historis."""
    risk_level = "Medium"
    volatility = 0.0
    risk_score = 5

    if not historical_df.empty and len(historical_df) > 1:
        historical_df['close'] = pd.to_numeric(historical_df['close'], errors='coerce').dropna()
        if len(historical_df['close']) > 1:
            try:
                # Hitung volatilitas menggunakan standar deviasi dari log return
                historical_df['log_returns'] = ta.others.log_return(close=historical_df['close'])
                if not historical_df['log_returns'].empty:
                    volatility = historical_df['log_returns'].std() * (252**0.5) * 100 # Volatilitas tahunan dalam persentase

                    if volatility < 1.0: risk_level, risk_score = "Very Low", 1
                    elif volatility < 2.0: risk_level, risk_score = "Low", 2
                    elif volatility < 5.0: risk_level, risk_score = "Medium", 4
                    elif volatility < 10.0: risk_level, risk_score = "High", 7
                    else: risk_level, risk_score = "Very High", 9
            except Exception as e:
                print(f"Error menghitung volatilitas: {e}. Menggunakan nilai default.")

    return risk_level, volatility, risk_score

# --- LOGIKA UTAMA BOT ---
def run_crypto_analysis_bot():
    """Fungsi utama untuk menjalankan analisis kripto, fokus pada target utama."""
    print(f"\n--- Analisis Teknikal {TARGET_CRYPTO_SYMBOL} (Fokus) ---")
    print(f"Waktu Lokal: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # 1. Dapatkan Data Harga Saat Ini
    current_data = get_crypto_data(TARGET_CRYPTO_SYMBOL)
    if not current_data:
        print(f"Analisis untuk {TARGET_CRYPTO_SYMBOL} tidak dapat dilanjutkan karena data terkini gagal diambil.")
        return

    print(f"\n💰 Harga Semasa")
    print(f"{current_data['name']} ({current_data['symbol']}): ${current_data['price']:.6f}")
    print(f"Perubahan 24j: {current_data['percent_change_24h']:.2f}%")

    print(f"\n📊 Volume & Market Cap")
    print(f"Volume 24j: ${current_data['volume_24h']:.2f}")
    print(f"Market Cap: ${current_data['market_cap']:.2f}")

    # 2. Dapatkan Data Historis untuk Indikator Teknis dan Penilaian Risiko
    historical_df = get_historical_data(TARGET_CRYPTO_SYMBOL)

    print(f"\n📈 Petunjuk Teknikal")
    indicators = calculate_technical_indicators(historical_df)
    signal, confidence, recommendation_text, indicator_statuses = generate_trading_signals(indicators)

    # Cetak hasil indikator
    print(f"📊 RSI(5,3,3): {indicators.get('RSI', 'N/A'):.1f} ({indicator_statuses.get('RSI', 'N/A')})")
    print(f"📉 Stochastic(5): K={indicators.get('Stochastic_K', 'N/A'):.1f} / D={indicators.get('Stochastic_D', 'N/A'):.1f} ({indicator_statuses.get('Stochastic', 'N/A')})")
    print(f"📈 MACD: {indicators.get('MACD', 'N/A'):.4f} / Signal={indicators.get('MACD_Signal', 'N/A'):.4f} ({indicator_statuses.get('MACD', 'N/A')})")


    # 3. Sinyal Trading
    print(f"\n➡️ Trading Signals")
    print(f"💡 Signal: {signal}")
    print(f"🔑 Confidence: {confidence}%")
    print(f"📝 Recommendation: {recommendation_text}")

    # 4. Analisis Pasar
    print(f"\n🔥 Market Analysis")
    sentiment, score, momentum = perform_market_analysis(current_data['percent_change_24h'])
    print(f"💡 Market Sentiment: {sentiment}")
    print(f"📊 Score: {score:+.1f}")
    print(f"📈 Momentum: {momentum}")

    # 5. Penilaian Risiko
    print(f"\n✅ Risk Assessment")
    risk_level, volatility, risk_score = perform_risk_assessment(historical_df)
    print(f"🛡️ Risk Level: {risk_level}")
    print(f"⚠️ Volatility: {volatility:.1f}%")
    print(f"🔥 Risk Score: {risk_score}/10")

    print("\n--- Analisis Selesai ---")

# --- TITIK MASUK PROGRAM ---
if __name__ == "__main__":
    run_crypto_analysis_bot()

