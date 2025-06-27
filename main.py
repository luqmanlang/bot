import os
import time
import pandas as pd
import ta
import discord
from discord.ext import commands # Untuk command framework

# --- IMPORT LIBRARY coinmarketcap ---
try:
    from coinmarketcap import Market
except ImportError:
    print("------------------------------------------------------------------")
    print("ERROR: Pustaka 'python-coinmarketcap' tidak ditemukan.")
    print("Pastikan Anda sudah menginstal dengan 'pip install python-coinmarketcap'")
    print("dan file 'requirements.txt' Anda berisi 'python-coinmarketcap'.")
    print("------------------------------------------------------------------")
    exit(1)

# --- KONFIGURASI API KEY ---
CMC_PRO_API_KEY = os.environ.get("CMC_PRO_API_KEY")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")

if not CMC_PRO_API_KEY:
    print("------------------------------------------------------------------")
    print("ERROR: Variabel lingkungan 'CMC_PRO_API_KEY' tidak diatur.")
    print("Silakan atur API Key CoinMarketCap Anda sebagai environment variable.")
    print("Contoh: export CMC_PRO_API_KEY='INI_API_KEY_ANDA'")
    print("Atau di dashboard Render, tambahkan environment variable tersebut.")
    print("------------------------------------------------------------------")
    exit(1)

if not DISCORD_BOT_TOKEN:
    print("------------------------------------------------------------------")
    print("ERROR: Variabel lingkungan 'DISCORD_BOT_TOKEN' tidak diatur.")
    print("Silakan atur Discord Bot Token Anda sebagai environment variable.")
    print("Dapatkan dari Discord Developer Portal: discord.com/developers/applications")
    print("------------------------------------------------------------------")
    exit(1)

# --- INISIALISASI KLIEN CMC API ---
try:
    coinmarketcap_client = Market(CMC_PRO_API_KEY)
except Exception as e:
    print("------------------------------------------------------------------")
    print(f"ERROR: Gagal menginisialisasi klien CoinMarketCap: {e}")
    print("Periksa kembali API Key Anda dan koneksi jaringan.")
    print("------------------------------------------------------------------")
    exit(1)

# --- KONFIGURASI BOT DISCORD ---
# Mengaktifkan intents yang diperlukan, terutama Message Content Intent
intents = discord.Intents.default()
intents.message_content = True # PENTING: Aktifkan ini di Discord Developer Portal juga!
intents.messages = True
intents.guilds = True

# Prefix untuk perintah bot, contoh: !analisis
bot = commands.Bot(command_prefix='!', intents=intents)

# --- KONFIGURASI ANALISIS KRIPTO ---
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
    # Untuk mendapatkan data historis yang valid, Anda kemungkinan besar membutuhkan
    # paket CoinMarketCap API berbayar atau menggunakan sumber data lain (misalnya ccxt).
    # Placeholder ini berfungsi untuk demonstrasi TA.
    num_points = limit
    data = {
        'open': [100 + i for i in range(num_points)],
        'high': [102 + i for i in range(num_points)],
        'low': [99 + i for i in range(num_points)],
        'close': [101 + i + (i % 5) * 0.5 - (i % 7) * 0.2 for i in range(num_points)],
        'volume': [1000 + i * 10 for i in range(num_points)]
    }
    df = pd.DataFrame(data)
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    df['timestamp'] = pd.to_datetime(pd.date_range(end=end_time_str, periods=len(df), freq='5min'))
    df = df.set_index('timestamp').sort_index()
    return df

# --- FUNGSI INDIKATOR TEKNIKAL ---
def calculate_technical_indicators(df):
    """Menghitung indikator teknikal dari DataFrame historis."""
    if df.empty or len(df) < 20:
        return {}
    indicators = {}
    try: indicators["RSI"] = ta.momentum.RSIIndicator(close=df['close'], window=5).rsi().iloc[-1]
    except Exception: indicators["RSI"] = None
    try:
        stoch_oscillator = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=5, smooth_window=3)
        indicators["Stochastic_K"] = stoch_oscillator.stoch().iloc[-1]
        indicators["Stochastic_D"] = stoch_oscillator.stoch_signal().iloc[-1]
    except Exception: indicators["Stochastic_K"], indicators["Stochastic_D"] = None, None
    try:
        macd_indicator = ta.trend.MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)
        indicators["MACD"] = macd_indicator.macd().iloc[-1]
        indicators["MACD_Signal"] = macd_indicator.macd_signal().iloc[-1]
    except Exception: indicators["MACD"], indicators["MACD_Signal"] = None, None
    return indicators

# --- FUNGSI Sinyal Trading ---
def generate_trading_signals(indicators):
    """Menghasilkan sinyal trading berdasarkan indikator teknikal."""
    signal = "HOLD"
    confidence = 0
    indicator_statuses = {"RSI": "Neutral", "Stochastic": "Neutral", "MACD": "Neutral"}

    rsi = indicators.get("RSI")
    stoch_k = indicators.get("Stochastic_K")
    stoch_d = indicators.get("Stochastic_D")
    macd = indicators.get("MACD")
    macd_signal = indicators.get("MACD_Signal")

    if rsi is not None:
        if rsi < 30: indicator_statuses["RSI"], confidence = "Oversold (Potensi Beli)", confidence + 20
        elif rsi > 70: indicator_statuses["RSI"], confidence = "Overbought (Potensi Jual)", confidence + 20

    if stoch_k is not None and stoch_d is not None:
        if stoch_k < 20 and stoch_k > stoch_d: indicator_statuses["Stochastic"], confidence = "Bullish Crossover (Beli Kuat)", confidence + 35
        elif stoch_k > 80 and stoch_k < stoch_d: indicator_statuses["Stochastic"], confidence = "Bearish Crossover (Jual Kuat)", confidence + 35
        elif stoch_k > 80: indicator_statuses["Stochastic"], confidence = "Overbought", confidence + 10
        elif stoch_k < 20: indicator_statuses["Stochastic"], confidence = "Oversold", confidence + 10

    if macd is not None and macd_signal is not None:
        if macd > macd_signal: indicator_statuses["MACD"], confidence = "Bullish Crossover", confidence + 25
        elif macd < macd_signal: indicator_statuses["MACD"], confidence = "Bearish Crossover", confidence + 25

    buy_votes, sell_votes = 0, 0
    if "Oversold" in indicator_statuses["RSI"]: buy_votes += 1
    if "Overbought" in indicator_statuses["RSI"]: sell_votes += 1
    if "Bullish Crossover" in indicator_statuses["Stochastic"] or "Oversold" in indicator_statuses["Stochastic"]: buy_votes += 1
    if "Bearish Crossover" in indicator_statuses["Stochastic"] or "Overbought" in indicator_statuses["Stochastic"]: sell_votes += 1
    if "Bullish Crossover" in indicator_statuses["MACD"]: buy_votes += 1
    if "Bearish Crossover" in indicator_statuses["MACD"]: sell_votes += 1

    if buy_votes >= 2: signal = "BUY"
    elif sell_votes >= 2: signal = "SELL"
    else: signal = "HOLD"

    confidence = min(confidence, 100)
    recommendation_text = f"Disyorkan untuk {signal} dengan keyakinan {confidence}%."
    if confidence < 60 and signal != "HOLD":
        recommendation_text += " Tunggu signal yang lebih jelas."
        signal = "HOLD"
    return signal, confidence, recommendation_text, indicator_statuses

# --- ANALISIS PASAR ---
def perform_market_analysis(price_change_24h):
    """Melakukan analisis sentimen pasar sederhana."""
    sentiment, score, momentum = "Neutral", 0.0, "Neutral"
    if price_change_24h > 5: sentiment, score, momentum = "Bullish", 2.0, "Strong Up"
    elif price_change_24h > 0: sentiment, score, momentum = "Slightly Bullish", 1.0, "Up"
    elif price_change_24h < -5: sentiment, score, momentum = "Bearish", -2.0, "Strong Down"
    elif price_change_24h < 0: sentiment, score, momentum = "Slightly Bearish", -1.0, "Down"
    return sentiment, score, momentum

# --- PENILAIAN RISIKO ---
def perform_risk_assessment(historical_df):
    """Melakukan penilaian risiko sederhana berdasarkan volatilitas historis."""
    risk_level, volatility, risk_score = "Medium", 0.0, 5
    if not historical_df.empty and len(historical_df) > 1:
        historical_df['close'] = pd.to_numeric(historical_df['close'], errors='coerce').dropna()
        if len(historical_df['close']) > 1:
            try:
                historical_df['log_returns'] = ta.others.log_return(close=historical_df['close'])
                if not historical_df['log_returns'].empty:
                    volatility = historical_df['log_returns'].std() * (252**0.5) * 100
                    if volatility < 1.0: risk_level, risk_score = "Very Low", 1
                    elif volatility < 2.0: risk_level, risk_score = "Low", 2
                    elif volatility < 5.0: risk_level, risk_score = "Medium", 4
                    elif volatility < 10.0: risk_level, risk_score = "High", 7
                    else: risk_level, risk_score = "Very High", 9
            except Exception as e: print(f"Error menghitung volatilitas: {e}. Menggunakan nilai default.")
    return risk_level, volatility, risk_score

# --- LOGIKA ANALISIS UTAMA (Fungsi Asynchronous untuk Discord) ---
async def get_full_analysis(symbol_to_analyze):
    """
    Menjalankan analisis lengkap untuk simbol kripto dan mengembalikan string laporan.
    Dibuat async karena akan dipanggil dari command Discord.
    """
    report = [f"--- Analisis Teknikal {symbol_to_analyze} ({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}) ---"]

    current_data = get_crypto_data(symbol_to_analyze)
    if not current_data:
        report.append(f"Tidak dapat mengambil data semasa untuk {symbol_to_analyze}. Sila cuba lagi nanti.")
        return "\n".join(report)

    report.append(f"\n💰 Harga Semasa")
    report.append(f"{current_data['name']} ({current_data['symbol']}): ${current_data['price']:.6f}")
    report.append(f"Perubahan 24j: {current_data['percent_change_24h']:.2f}%")
    report.append(f"Volume 24j: ${current_data['volume_24h']:.2f}")
    report.append(f"Market Cap: ${current_data['market_cap']:.2f}")

    historical_df = get_historical_data(symbol_to_analyze)
    indicators = calculate_technical_indicators(historical_df)
    signal, confidence, recommendation_text, indicator_statuses = generate_trading_signals(indicators)

    report.append(f"\n📈 Petunjuk Teknikal")
    report.append(f"📊 RSI(5,3,3): {indicators.get('RSI', 'N/A'):.1f} ({indicator_statuses.get('RSI', 'N/A')})")
    report.append(f"📉 Stochastic(5): K={indicators.get('Stochastic_K', 'N/A'):.1f} / D={indicators.get('Stochastic_D', 'N/A'):.1f} ({indicator_statuses.get('Stochastic', 'N/A')})")
    report.append(f"📈 MACD: {indicators.get('MACD', 'N/A'):.4f} / Signal={indicators.get('MACD_Signal', 'N/A'):.4f} ({indicator_statuses.get('MACD', 'N/A')})")

    report.append(f"\n➡️ Trading Signals")
    report.append(f"💡 Signal: {signal}")
    report.append(f"🔑 Confidence: {confidence}%")
    report.append(f"📝 Recommendation: {recommendation_text}")

    sentiment, score, momentum = perform_market_analysis(current_data['percent_change_24h'])
    report.append(f"\n🔥 Market Analysis")
    report.append(f"💡 Market Sentiment: {sentiment}")
    report.append(f"📊 Score: {score:+.1f}")
    report.append(f"📈 Momentum: {momentum}")

    risk_level, volatility, risk_score = perform_risk_assessment(historical_df)
    report.append(f"\n✅ Risk Assessment")
    report.append(f"🛡️ Risk Level: {risk_level}")
    report.append(f"⚠️ Volatility: {volatility:.1f}%")
    report.append(f"🔥 Risk Score: {risk_score}/10")

    report.append("\n--- Analisis Selesai ---")
    return "\n".join(report)

# --- EVENT HANDLERS DISCORD ---
@bot.event
async def on_ready():
    """Event yang dipicu saat bot berhasil terhubung ke Discord."""
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    print(f'Bot siap menerima perintah dengan prefix: {bot.command_prefix}')
    print('----------------------------------------------------')
    # Anda bisa mengatur status bot di sini
    await bot.change_presence(activity=discord.Game(name=f"Analisis {TARGET_CRYPTO_SYMBOL} | !analisis"))

@bot.event
async def on_command_error(ctx, error):
    """Event yang dipicu saat terjadi error pada perintah."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f"Maaf, perintah '{ctx.invoked_with}' tidak ditemukan. Gunakan `!analisis`.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Kesalahan argumen: {error}. Contoh: `!analisis BTC`.")
    else:
        print(f"Error pada perintah {ctx.command}: {error}")
        await ctx.send(f"Terjadi kesalahan saat menjalankan perintah: {error}")

# --- PERINTAH BOT DISCORD ---
@bot.command(name='analisis', help='Melakukan analisis teknikal untuk kripto tertentu. Contoh: !analisis BTC')
async def analisis(ctx, symbol: str = None):
    """
    Perintah untuk melakukan analisis teknikal kripto.
    Jika tidak ada simbol yang diberikan, akan menganalisis TARGET_CRYPTO_SYMBOL.
    """
    await ctx.send(f"Sedang menganalisis {symbol if symbol else TARGET_CRYPTO_SYMBOL}...")

    # Tentukan simbol yang akan dianalisis
    target_symbol = symbol.upper() if symbol else TARGET_CRYPTO_SYMBOL

    # Jalankan analisis dan kirim hasilnya
    analysis_report = await get_full_analysis(target_symbol)

    # Kirim laporan dalam block kode untuk keterbacaan yang lebih baik
    # Batasi panjang pesan Discord (maks 2000 karakter)
    if len(analysis_report) > 1990:
        # Potong laporan jika terlalu panjang
        analysis_report = analysis_report[:1900] + "\n... (Laporan terlalu panjang, dipotong)"
    await ctx.send(f"```\n{analysis_report}\n```")

# --- TITIK MASUK PROGRAM (Menjalankan Bot Discord) ---
if __name__ == "__main__":
    print("Memulai bot Discord...")
    # Token diambil dari environment variable DISCORD_BOT_TOKEN
    bot.run(DISCORD_BOT_TOKEN)

