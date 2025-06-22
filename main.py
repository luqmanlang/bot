import os
import discord
from discord.ext import commands, tasks
import aiohttp
import asyncio
import datetime
import logging
import math
import statistics
from datetime import timezone # Import timezone for utcnow

# --- === KONFIGURASI UMUM === ---
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
CMC_API_KEY = os.getenv("CMC_API_KEY") # NEW: CoinMarketCap API Key
ALERT_CHANNEL_ID = int(os.getenv("BITCOIN_ALERT_CHANNEL_ID", "123456789012345678")) # Gantikan dengan ID saluran sebenar
BOT_PREFIX = '!'
CMC_API_BASE = "https://pro-api.coinmarketcap.com" # CMC Pro API base URL

# Ambang Alert (Peratusan perubahan dalam 24 jam)
ALERT_THRESHOLD_MICRO = 0.5
ALERT_THRESHOLD_MACRO = 3.0
ALERT_THRESHOLD_CRITICAL = 7.0
MIN_PRICE_CHANGE_FOR_ALERT = 0.1 # Minimum perubahan mutlak harga untuk mencetuskan alert

# Konfigurasi Indikator
RSI_PERIOD = 14
STOCHASTIC_PERIOD = 14
ATR_PERIOD = 14
MA_SHORT_PERIOD = 10
MA_LONG_PERIOD = 50
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2

# Global untuk cooldown alert
LAST_ALERT_TIME = None
ALERT_COOLDOWN_HOURS = 2 # Alert yang sama tidak akan dicetuskan dalam 2 jam (kerana panggilan API setiap jam)

# --- === SETUP LOGGING === ---
if not os.path.exists('logs'):
    os.makedirs('logs')
LOG_FILE_PATH = 'logs/bot.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BitcoinSentinelAdvance")

# --- === KELAS UTAMA BOT (Menggunakan Cog untuk Pengurusan State dan Logik) === ---
class BitcoinSentinelAdvance(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.btc_price_history = [] # Menyimpan harga penutup sahaja untuk kemudahan (untuk indikator tertentu)
        self.btc_ohlc_history = [] # Menyimpan (high, low, close) sebenar untuk ATR/BB
        self.last_btc_data = {'price': 0.0, 'change_24hr': 0.0}
        self.last_eth_data = {'price': 0.0, 'change_24hr': 0.0}
        self.alert_channel = None

    # --- === UTILITI DALAMAN === ---
    def _now_utc(self):
        return datetime.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    def _log_error(self, e, label="ERROR"):
        logger.error(f"[{label}] {e}", exc_info=True)

    def _get_trend_icon(self, change):
        return "📈" if change > 0 else "📉" if change < 0 else "➖"

    def _get_sentiment(self, change):
        if change > 3: return "🐂 Bullish"
        if change < -3: return "🐻 Bearish"
        return "⚖️ Neutral"

    def _risk_level(self, change):
        abs_change = abs(change)
        if abs_change >= ALERT_THRESHOLD_CRITICAL: return "🔴 Tinggi"
        if abs_change >= ALERT_THRESHOLD_MACRO: return "🟠 Sederhana"
        return "🟢 Rendah"

    # --- === FUNGSI KALKULASI INDIKATOR TEKNIKAL === ---

    def _calc_rsi(self, prices, period=RSI_PERIOD):
        if len(prices) < period + 1: return None
        
        gains = [0.0] * len(prices)
        losses = [0.0] * len(prices)
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0: gains[i] = change
            else: losses[i] = abs(change)

        avg_gain = sum(gains[1:period+1]) / period
        avg_loss = sum(losses[1:period+1]) / period

        rsi_values = []

        if avg_loss == 0:
            rs = 100 if avg_gain > 0 else (0 if avg_gain == 0 else 0) # Handle 0 gain, 0 loss
        else:
            rs = avg_gain / avg_loss
        
        rsi_values.append(100 - (100 / (1 + rs)) if rs is not None else None)

        for i in range(period + 1, len(prices)):
            current_gain = gains[i]
            current_loss = losses[i]
            
            avg_gain = ((avg_gain * (period - 1)) + current_gain) / period
            avg_loss = ((avg_loss * (period - 1)) + current_loss) / period
            
            if avg_loss == 0:
                rs = 100 if avg_gain > 0 else (0 if avg_gain == 0 else 0)
            else:
                rs = avg_gain / avg_loss
            
            rsi_values.append(100 - (100 / (1 + rs)) if rs is not None else None)
        
        return rsi_values[-1] if rsi_values else None


    def _calc_stochastic(self, prices, period=STOCHASTIC_PERIOD):
        if len(prices) < period: return None
        
        window_prices = prices[-period:]
        if not window_prices: return None 
        
        lowest_low = min(window_prices)
        highest_high = max(window_prices)
        current_price = prices[-1]

        if (highest_high - lowest_low) == 0:
            return 50.0 # Neutral if no price change or constant price
        
        k_percent = ((current_price - lowest_low) / (highest_high - lowest_low)) * 100
        return k_percent

    def _calc_ma(self, prices, period):
        if len(prices) < period: return None
        return statistics.mean(prices[-period:])

    def _calc_ema(self, prices, period):
        if len(prices) < period: return None
        
        ema_values = []
        smoothing_factor = 2 / (period + 1)
        
        ema_values.append(statistics.mean(prices[:period]))
        
        for i in range(period, len(prices)):
            current_price = prices[i]
            prev_ema = ema_values[-1]
            current_ema = (current_price - prev_ema) * smoothing_factor + prev_ema
            ema_values.append(current_ema)
            
        return ema_values[-1] if ema_values else None

    def _calc_macd(self, prices, fast_period=MACD_FAST_PERIOD, slow_period=MACD_SLOW_PERIOD, signal_period=MACD_SIGNAL_PERIOD):
        # Need enough prices for both slow EMA and then signal line EMA
        if len(prices) < slow_period + signal_period: return None 
        
        # Calculate full EMA series for MACD line and then Signal Line
        fast_emas_series = []
        slow_emas_series = []

        smoothing_fast = 2 / (fast_period + 1)
        smoothing_slow = 2 / (slow_period + 1)

        # Initial EMAs
        if len(prices) >= fast_period:
            fast_emas_series.append(statistics.mean(prices[:fast_period]))
        else: # Not enough data for initial fast EMA
            return None 

        if len(prices) >= slow_period:
            slow_emas_series.append(statistics.mean(prices[:slow_period]))
        else: # Not enough data for initial slow EMA
            return None

        # Calculate EMA series
        for i in range(1, len(prices)):
            # Fast EMA series
            if i >= fast_period:
                current_fast_ema = (prices[i] - fast_emas_series[-1]) * smoothing_fast + fast_emas_series[-1]
                fast_emas_series.append(current_fast_ema)
            else: # Pad with previous valid EMA if not enough data yet
                fast_emas_series.append(fast_emas_series[-1] if fast_emas_series else None)

            # Slow EMA series
            if i >= slow_period:
                current_slow_ema = (prices[i] - slow_emas_series[-1]) * smoothing_slow + slow_emas_series[-1]
                slow_emas_series.append(current_slow_ema)
            else: # Pad with previous valid EMA
                slow_emas_series.append(slow_emas_series[-1] if slow_emas_series else None)

        # Align series to start where both fast and slow EMAs are valid
        start_index = max(fast_period - 1, slow_period - 1)
        if start_index >= len(fast_emas_series) or start_index >= len(slow_emas_series):
            return None # Not enough valid EMAs

        # Calculate MACD Line
        macd_line_series = []
        for i in range(start_index, len(prices)):
            if fast_emas_series[i] is not None and slow_emas_series[i] is not None:
                macd_line_series.append(fast_emas_series[i] - slow_emas_series[i])
            else:
                macd_line_series.append(None)
        
        # Filter out None values to prevent issues in signal line calculation
        macd_line_series = [val for val in macd_line_series if val is not None]

        if len(macd_line_series) < signal_period: return None

        # Calculate Signal Line (EMA of MACD Line)
        signal_line_series = []
        smoothing_signal = 2 / (signal_period + 1)

        signal_line_series.append(statistics.mean(macd_line_series[:signal_period]))

        for i in range(signal_period, len(macd_line_series)):
            current_signal = (macd_line_series[i] - signal_line_series[-1]) * smoothing_signal + signal_line_series[-1]
            signal_line_series.append(current_signal)

        # Histogram is MACD Line - Signal Line
        histogram = macd_line_series[-1] - signal_line_series[-1] if macd_line_series and signal_line_series else None
        
        return {
            'macd_line': macd_line_series[-1] if macd_line_series else None,
            'signal_line': signal_line_series[-1] if signal_line_series else None,
            'histogram': histogram
        }
    
    def _calc_bollinger_bands(self, ohlc_data, period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD_DEV):
        # Expects a list of (high, low, close) tuples, we use close for BB
        if len(ohlc_data) < period: return None

        prices = [d[2] for d in ohlc_data] # Extract close prices
        window_prices = prices[-period:]
        if len(window_prices) < period: return None # Should be guaranteed by outer check
        
        middle_band = statistics.mean(window_prices)
        
        if len(window_prices) < 2: 
            std_dev_value = 0
        else:
            std_dev_value = statistics.pstdev(window_prices)

        upper_band = middle_band + (std_dev_value * std_dev)
        lower_band = middle_band - (std_dev_value * std_dev)
        
        return {
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band
        }

    def _calc_atr(self, ohlc_data, period=ATR_PERIOD):
        # Expects a list of (high, low, close) tuples
        if len(ohlc_data) < period + 1: return None
        
        true_ranges = []
        for i in range(1, len(ohlc_data)):
            prev_close = ohlc_data[i-1][2]
            current_high = ohlc_data[i][0]
            current_low = ohlc_data[i][1]
            
            tr1 = current_high - current_low
            tr2 = abs(current_high - prev_close)
            tr3 = abs(current_low - prev_close)
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period: return None 
        atr_value = sum(true_ranges[:period]) / period # Initial SMA

        for i in range(period, len(true_ranges)):
            atr_value = ((atr_value * (period - 1)) + true_ranges[i]) / period
            
        return atr_value

    # --- === LOGIK PENGAMBILAN DATA (CMC API) === ---
    async def _fetch_json_cmc(self, url, params=None):
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': CMC_API_KEY,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as resp:
                    resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                    return await resp.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                self._log_error(f"CMC API Rate Limit Reached! ({e.status})", "CMC_API_ERROR")
                await asyncio.sleep(60) # Wait a minute before retrying
                return None
            else:
                self._log_error(f"CMC API HTTP Error: {e.status} - {e.message} from {url}", "CMC_API_ERROR")
            return None
        except aiohttp.ClientError as e:
            self._log_error(f"HTTP Client Error: {e} from {url}", "CMC_API_FETCH")
            return None
        except Exception as e:
            self._log_error(f"Unexpected error fetching {url}: {e}", "CMC_API_FETCH")
            return None

    async def _get_current_price_data(self):
        # Using CMC /quotes/latest for current price and 24h change
        url = f"{CMC_API_BASE}/v2/cryptocurrency/quotes/latest"
        params = {'symbol': 'BTC,ETH', 'convert': 'USD'}
        data = await self._fetch_json_cmc(url, params)
        
        if data and data.get('data'):
            btc_data = data['data'].get('BTC', [{}])[0].get('quote', {}).get('USD', {})
            eth_data = data['data'].get('ETH', [{}])[0].get('quote', {}).get('USD', {})

            return {
                'btc': btc_data.get('price'),
                'btc_change': btc_data.get('percent_change_24h'),
                'eth': eth_data.get('price'),
                'eth_change': eth_data.get('percent_change_24h')
            }
        return None

    async def _get_historical_market_data(self, symbol='BTC', days=90, interval='daily'):
        # Using CMC /ohlcv/historical for OHLC data
        # CMC's hourly data is for up to 7 days for free tier
        # For longer periods, 'daily' interval is needed.
        
        url = f"{CMC_API_BASE}/v2/cryptocurrency/ohlcv/historical"
        
        # Calculate start and end time
        end_time = datetime.datetime.now(timezone.utc)
        start_time = end_time - datetime.timedelta(days=days)

        # Format times for CMC API
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())

        params = {
            'symbol': symbol,
            'time_period': f"24h" if interval == 'daily' else f"hourly", # This seems to be the correct param for interval
            'time_start': start_time.isoformat(timespec='seconds') + "Z",
            'time_end': end_time.isoformat(timespec='seconds') + "Z",
            'interval': interval # The actual interval parameter
        }
        
        # CMC Free Tier limitation: 1 day historical for hourly. Up to 90 days for daily.
        # Adjusted parameter mapping for different intervals based on free tier.
        if interval == 'hourly' and days > 7: # CMC free tier limits hourly to max 7 days.
            logger.warning(f"Meminta data {symbol} hourly untuk {days} hari, tetapi CMC API mungkin terhad kepada 7 hari untuk hourly. Menggunakan {min(days, 7)} hari.")
            start_time_hourly_limit = end_time - datetime.timedelta(days=min(days, 7))
            params['time_start'] = start_time_hourly_limit.isoformat(timespec='seconds') + "Z"
            # No change to 'interval', it will still request 'hourly' but for shorter period.

        data = await self._fetch_json_cmc(url, params)

        prices = []
        ohlc_data = [] # (high, low, close)
        volumes = []

        if data and data.get('data') and data['data'].get(symbol) and data['data'][symbol].get('quotes'):
            quotes = data['data'][symbol]['quotes']
            for quote in quotes:
                ohlcv = quote.get('ohlcv', {})
                if ohlcv:
                    open_price = ohlcv.get('open')
                    high_price = ohlcv.get('high')
                    low_price = ohlcv.get('low')
                    close_price = ohlcv.get('close')
                    volume = ohlcv.get('volume')

                    if all(v is not None for v in [open_price, high_price, low_price, close_price, volume]):
                        prices.append(close_price)
                        ohlc_data.append((high_price, low_price, close_price))
                        volumes.append(volume)
            logger.info(f"Berjaya memuatkan {len(prices)} titik data sejarah {symbol} ({interval}, {days} hari) dari CMC.")
            return {
                'prices': prices,
                'volumes': volumes,
                'high_low_close': ohlc_data
            }
        logger.warning(f"Gagal memuatkan data sejarah {symbol} ({interval}, {days} hari) dari CMC.")
        return None

    # --- === ANALISIS & GENERASI LAPORAN === ---

    async def _perform_detailed_analysis(self, coin_id: str, days: int, interval: str = 'daily'):
        historical_data = await self._get_historical_market_data(coin_id, days, interval)
        current_data = await self._get_current_price_data()

        if not historical_data or not current_data or not historical_data['prices']:
            return "❌ Gagal mengambil data yang diperlukan untuk laporan analisis. Sila cuba lagi."

        prices = historical_data['prices']
        ohlc_data = historical_data['high_low_close']
        volumes = historical_data['volumes']

        # Ensure enough data points for all indicators
        min_data_needed = max(RSI_PERIOD, STOCHASTIC_PERIOD, MA_LONG_PERIOD, ATR_PERIOD, BOLLINGER_PERIOD, MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD)
        if len(prices) < min_data_needed:
            return f"❌ Tidak cukup data ({len(prices)} titik) untuk analisis {days} hari ({interval} interval). Perlukan sekurang-kurangnya {min_data_needed} titik data. Cuba jangka masa yang lebih panjang atau 'daily' interval."

        # Indikator
        rsi = self._calc_rsi(prices)
        stoch = self._calc_stochastic(prices)
        short_ma = self._calc_ma(prices, MA_SHORT_PERIOD)
        long_ma = self._calc_ma(prices, MA_LONG_PERIOD)
        macd_results = self._calc_macd(prices)
        bollinger_bands = self._calc_bollinger_bands(ohlc_data) # Use actual OHLC
        atr = self._calc_atr(ohlc_data, ATR_PERIOD) # Use actual OHLC
        
        macd_line = macd_results['macd_line'] if macd_results else None
        signal_line = macd_results['signal_line'] if macd_results else None
        histogram = macd_results['histogram'] if macd_results else None

        # Analisis Umum
        start_price = prices[0]
        end_price = prices[-1]
        percent_change = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0
        high_price = max([d[0] for d in ohlc_data]) # Max of actual Highs
        low_price = min([d[1] for d in ohlc_data]) # Min of actual Lows
        avg_volume = statistics.mean(volumes) if volumes else 0
        current_volume = volumes[-1] if volumes else 0

        # Simpulkan Signal
        signal_summary = []
        def is_valid_num(val): return val is not None and not math.isnan(val) and not math.isinf(val)

        if is_valid_num(rsi):
            if rsi > 70: signal_summary.append("RSI **Overbought** (Potensi pembalikan)")
            elif rsi < 30: signal_summary.append("RSI **Oversold** (Potensi pemulihan)")
            else: signal_summary.append("RSI Neutral")
        
        if is_valid_num(stoch):
            if stoch > 80: signal_summary.append("Stochastic **Overbought** (Amaran jualan)")
            elif stoch < 20: signal_summary.append("Stochastic **Oversold** (Amaran belian)")
            else: signal_summary.append("Stochastic Neutral")
        
        if is_valid_num(short_ma) and is_valid_num(long_ma):
            if short_ma > long_ma and prices[-1] > short_ma:
                signal_summary.append("MA Crossover **Bullish** (Uptrend Kukuh)")
            elif short_ma < long_ma and prices[-1] < short_ma:
                signal_summary.append("MA Crossover **Bearish** (Downtrend Kukuh)")
            else:
                signal_summary.append("MA Neutral/Sideways")
        
        if macd_results and is_valid_num(macd_line) and is_valid_num(signal_line):
            if macd_line > signal_line and (histogram is None or histogram > 0): signal_summary.append("MACD Bullish Crossover (Momentum Meningkat)")
            elif macd_line < signal_line and (histogram is None or histogram < 0): signal_summary.append("MACD Bearish Crossover (Momentum Menurun)")
            elif abs(macd_line - signal_line) < (end_price * 0.0001): signal_summary.append("MACD Neutral (Sideways)")
            
        if bollinger_bands and all(is_valid_num(v) for v in bollinger_bands.values()):
            current_price = prices[-1]
            if current_price > bollinger_bands['upper']: signal_summary.append("Bollinger Bands: **Overbought** (Harga di atas Upper Band)")
            elif current_price < bollinger_bands['lower']: signal_summary.append("Bollinger Bands: **Oversold** (Harga di bawah Lower Band)")
            elif abs(bollinger_bands['upper'] - bollinger_bands['lower']) < end_price * 0.03: 
                signal_summary.append("Bollinger Bands: **Squeeze** (Volatiliti rendah, potensi breakout)")
            
        # "Whale Alert" yang lebih canggih (berdasarkan pergerakan harga relatif dan volume)
        whale_activity = ""
        if current_data['btc_change'] is not None and abs(current_data['btc_change']) > 5 and current_volume > avg_volume * 1.5:
            whale_activity = "🐳 **Aktiviti Whale Terkesan:** Pergerakan harga besar dengan volume luar biasa!"

        # Ringkasan AI John & Alpha
        ai_john_insight = self._generate_ai_insight(current_data['btc_change'], rsi, stoch, macd_results, bollinger_bands, prices[-1], "JohnAI")
        ai_alpha_insight = self._generate_ai_insight(current_data['btc_change'], rsi, stoch, macd_results, bollinger_bands, prices[-1], "AlphaAI")

        # Bangunkan Embed
        report_title = f"📈 Laporan Analisis Mendalam {coin_id.upper()} ({days} Hari - {interval.capitalize()}) 📉"
        report_description = f"**Gambaran Keseluruhan:** Harga dari `${start_price:,.2f}` ke `${end_price:,.2f}`\n" \
                             f"Perubahan: **{percent_change:+.2f}%** | High: **${high_price:,.2f}** | Low: **${low_price:,.2f}**"

        fields = []
        
        tech_indicator_string = (
            f"RSI({RSI_PERIOD}): `{rsi:.2f}`\n"
            f"Stochastic({STOCHASTIC_PERIOD}): `{stoch:.2f}`\n"
            f"MA({MA_SHORT_PERIOD})/MA({MA_LONG_PERIOD}): `${short_ma:,.2f}`/`${long_ma:,.2f}`\n"
            f"MACD: `{macd_line:.2f}` (Signal: `{signal_line:.2f}` | Hist: `{histogram:.2f}`)\n"
            f"Bollinger Bands: (Upper: `${bollinger_bands['upper']:,.2f}` | Middle: `${bollinger_bands['middle']:,.2f}` | Lower: `${bollinger_bands['lower']:,.2f}`)\n"
            f"ATR({ATR_PERIOD}): `${atr:.2f}` (Volatiliti)" 
            if all(is_valid_num(v) for v in [rsi, stoch, short_ma, long_ma, macd_line, signal_line, histogram, atr]) and bollinger_bands and all(is_valid_num(v) for v in bollinger_bands.values()) else "Data tidak mencukupi atau ralat kalkulasi."
        )
        fields.append(("Indikator Teknikal", tech_indicator_string, False))
        
        fields.append(("Ringkasan Isyarat", "\n".join(signal_summary) if signal_summary else "Tiada isyarat jelas.", False))
        fields.append(("Volume & Whale Alert", f"Volume Semasa: `{current_volume:,.0f}` (Avg: `{avg_volume:,.0f}`)\n{whale_activity}", False))
        fields.append(("Pandangan AI John (Trend Analyst)", ai_john_insight, False))
        fields.append(("Pandangan AlphaAI (Risk & Volatility Analyst)", ai_alpha_insight, False))
        
        embed_color = discord.Color.green() if percent_change >= 0 else discord.Color.red()
        
        return self._build_embed(report_title, report_description, fields, embed_color)

    def _generate_ai_insight(self, change: float, rsi: float, stoch: float, macd_results: dict, bollinger_bands: dict, current_price: float, ai_name: str):
        sentiment = []
        
        def is_valid_num(val):
            return val is not None and not math.isnan(val) and not math.isinf(val)

        # JohnAI: Trend-focused, more optimistic on confirmed trends
        if ai_name == "JohnAI":
            if is_valid_num(change) and change > ALERT_THRESHOLD_CRITICAL:
                sentiment.append("🚀 **Momentum Beli Kuat:** Kenaikan tajam dikesan. Pasaran menunjukkan kekuatan.")
                if is_valid_num(rsi) and rsi < 80: sentiment.append("RSI belum overbought sepenuhnya, ada ruang untuk kenaikan.")
                if macd_results and is_valid_num(macd_results['macd_line']) and is_valid_num(macd_results['signal_line']) and macd_results['macd_line'] > macd_results['signal_line']:
                    sentiment.append("MACD mengesahkan trend menaik.")
            elif is_valid_num(change) and change < -ALERT_THRESHOLD_CRITICAL:
                sentiment.append("📉 **Penurunan Tajam:** Awas, tekanan jualan tinggi.")
                if is_valid_num(rsi) and rsi > 20: sentiment.append("RSI belum oversold teruk, mungkin ada penurunan lanjut.")
                if macd_results and is_valid_num(macd_results['macd_line']) and is_valid_num(macd_results['signal_line']) and macd_results['macd_line'] < macd_results['signal_line']:
                    sentiment.append("MACD mengesahkan trend menurun.")
            elif is_valid_num(change) and change > ALERT_THRESHOLD_MACRO:
                sentiment.append("📈 **Trend Menaik Jelas:** Pasaran dalam fasa optimis.")
            elif is_valid_num(change) and change < -ALERT_THRESHOLD_MACRO:
                sentiment.append("📊 **Trend Menurun Jelas:** Hati-hati, pasaran dalam fasa pesimis.")
            else:
                sentiment.append("🔄 **Pasaran Sideways/Stabil:** Tiada trend jelas, sesuai untuk *range trading* atau *DCA*.")
            
            if bollinger_bands and is_valid_num(bollinger_bands['upper']) and is_valid_num(bollinger_bands['lower']):
                band_width = bollinger_bands['upper'] - bollinger_bands['lower']
                if is_valid_num(current_price) and band_width < current_price * 0.02: 
                    sentiment.append("Bollinger Bands menunjukkan **squeeze**, potensi *breakout* semakin dekat.")

        # AlphaAI: Risk-focused, more cautious and highlights potential reversals/volatility
        elif ai_name == "AlphaAI":
            if is_valid_num(change) and change > ALERT_THRESHOLD_CRITICAL:
                sentiment.append("⚠️ **Amaran Overbought!** Harga naik mendadak, kemungkinan pembetulan pantas.")
                if is_valid_num(rsi) and rsi > 70: sentiment.append("RSI yang sangat tinggi menguatkan amaran ini.")
                if bollinger_bands and is_valid_num(bollinger_bands['upper']) and is_valid_num(current_price) and current_price > bollinger_bands['upper']:
                    sentiment.append("Harga di atas Bollinger Upper Band, mungkin tidak mampan.")
            elif is_valid_num(change) and change < -ALERT_THRESHOLD_CRITICAL:
                sentiment.append("🚨 **Amaran Oversold!** Jangan terburu-buru menangkap 'falling knife'.")
                if is_valid_num(rsi) and rsi < 30: sentiment.append("RSI yang sangat rendah menunjukkan tekanan jualan melampau.")
                if bollinger_bands and is_valid_num(bollinger_bands['lower']) and is_valid_num(current_price) and current_price < bollinger_bands['lower']:
                    sentiment.append("Harga di bawah Bollinger Lower Band, mungkin akan melantun semula.")
            elif is_valid_num(change) and change > ALERT_THRESHOLD_MACRO:
                sentiment.append("🧠 **Sentimen Positif:** Namun, awasi *volume* dan *whale activity*. Elak *FOMO*.")
            elif is_valid_num(change) and change < -ALERT_THRESHOLD_MACRO:
                sentiment.append("📉 **Sentimen Negatif:** Prioritaskan pengurusan risiko. Elakkan *overtrading*.")
            else:
                sentiment.append("💤 **Volatiliti Rendah:** Masa sesuai untuk mengkaji semula strategi.")
                if bollinger_bands and is_valid_num(bollinger_bands['upper']) and is_valid_num(bollinger_bands['lower']):
                    band_width = bollinger_bands['upper'] - bollinger_bands['lower']
                    if is_valid_num(current_price) and band_width > current_price * 0.05: 
                        sentiment.append("Bollinger Bands lebar menunjukkan volatiliti tinggi. Berhati-hati.")

        return "\n".join(sentiment) if sentiment else "Analisis AI sedang menunggu data atau tiada isyarat jelas."


    # --- === PEMBINA EMBED DISCORD === ---
    def _build_embed(self, title, description, fields=None, color=discord.Color.blue(), footer_text="Analisis Bitcoin AI Sentinel | Data dari CoinMarketCap", timestamp=True):
        embed = discord.Embed(
            title=title,
            description=description,
            color=color
        )
        if fields:
            for name, value, inline in fields:
                if value and value.strip(): # Ensure value is not empty or just whitespace
                    embed.add_field(name=name, value=value, inline=inline)
        if footer_text:
            embed.set_footer(text=footer_text)
        if timestamp:
            embed.timestamp = datetime.datetime.utcnow()
        return embed

    # --- === TASKS LATAR BELAKANG === ---
    @tasks.loop(hours=1) # Diubah dari minutes=15 ke hours=1 untuk jimat kredit CMC
    async def monitor_bitcoin_price(self):
        global LAST_ALERT_TIME
        logger.info(f"Memulakan kitaran pemantauan harga Bitcoin ({self._now_utc()})...")
        current_data = await self._get_current_price_data()

        if not current_data or current_data['btc'] is None:
            logger.error("Gagal mengambil data Bitcoin semasa dari CMC. Melangkau semakan alert.")
            return

        btc_price = current_data['btc']
        btc_change = current_data['btc_change']
        eth_price = current_data['eth']
        eth_change = current_data['eth_change']

        if not self.alert_channel:
            self.alert_channel = self.bot.get_channel(ALERT_CHANNEL_ID)
            if not self.alert_channel:
                self._log_error(f"Saluran alert dengan ID {ALERT_CHANNEL_ID} tidak ditemui.", "ChannelError")
                return

        # Fetch actual OHLC for the latest hour from CMC for history if not done by startup
        # This will fetch a single candle. For longer history, on_ready handles it.
        latest_ohlcv_data = await self._get_historical_market_data(symbol='BTC', days=1, interval='hourly')
        
        if latest_ohlcv_data and latest_ohlcv_data['high_low_close']:
            # Take the very last OHLC entry as the most recent data point
            latest_ohlc_entry = latest_ohlcv_data['high_low_close'][-1]
            latest_close_price = latest_ohlc_entry[2]
            
            # Ensure price history is up-to-date with the latest *hourly* close
            if not self.btc_price_history or latest_close_price != self.btc_price_history[-1]:
                self.btc_price_history.append(latest_close_price)
                self.btc_ohlc_history.append(latest_ohlc_entry)
        else:
            logger.warning("Gagal mengambil data OHLCV terbaru dari CMC untuk mengemaskini sejarah. Menggunakan harga semasa.")
            # Fallback to current price if OHLCV for hour not available (less ideal for indicator history)
            self.btc_price_history.append(btc_price)
            # Simulate OHLC for this fallback point using current price
            self.btc_ohlc_history.append((btc_price * 1.005, btc_price * 0.995, btc_price))
            
        # Ensure history does not grow indefinitely, keep enough for all indicators
        max_history_needed = max(RSI_PERIOD, STOCHASTIC_PERIOD, ATR_PERIOD, MA_LONG_PERIOD, BOLLINGER_PERIOD, MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD) + 5 
        
        if len(self.btc_price_history) > max_history_needed:
            self.btc_price_history = self.btc_price_history[-max_history_needed:]
            self.btc_ohlc_history = self.btc_ohlc_history[-max_history_needed:]

        # Alert logic
        alert_type = "None"
        if abs(btc_change) >= ALERT_THRESHOLD_CRITICAL:
            alert_type = "Critical Shift"
        elif abs(btc_change) >= ALERT_THRESHOLD_MACRO:
            alert_type = "Macro Shift"
        elif abs(btc_change) >= ALERT_THRESHOLD_MICRO:
            alert_type = "Micro Fluctuation"

        current_time = datetime.datetime.now(timezone.utc)
        
        is_significant_price_change = abs(btc_price - self.last_btc_data['price']) >= MIN_PRICE_CHANGE_FOR_ALERT
        
        is_new_alert_type = (
            (alert_type == "Micro Fluctuation" and abs(self.last_btc_data['change_24hr']) < ALERT_THRESHOLD_MICRO) or
            (alert_type == "Macro Shift" and abs(self.last_btc_data['change_24hr']) < ALERT_THRESHOLD_MACRO) or
            (alert_type == "Critical Shift" and abs(self.last_btc_data['change_24hr']) < ALERT_THRESHOLD_CRITICAL)
        )

        should_send_alert = False
        if alert_type != "None" and (is_significant_price_change or is_new_alert_type):
            if LAST_ALERT_TIME:
                time_diff = (current_time - LAST_ALERT_TIME).total_seconds() / 3600 # dalam jam
                if time_diff >= ALERT_COOLDOWN_HOURS:
                    should_send_alert = True
                else:
                    logger.info(f"Dalam tempoh cooldown alert ({time_diff:.1f}/{ALERT_COOLDOWN_HOURS} jam). Melangkau alert.")
            else: 
                should_send_alert = True

        if should_send_alert:
            logger.warning(f"Mencetuskan alert {alert_type}.")
            
            # Recalculate indicators with the latest history
            # Ensure enough data points for calculation
            if len(self.btc_price_history) >= max_history_needed:
                rsi = self._calc_rsi(self.btc_price_history)
                stoch = self._calc_stochastic(self.btc_price_history)
                atr = self._calc_atr(self.btc_ohlc_history, ATR_PERIOD)
                short_ma = self._calc_ma(self.btc_price_history, MA_SHORT_PERIOD)
                long_ma = self._calc_ma(self.btc_price_history, MA_LONG_PERIOD)
                macd_results = self._calc_macd(self.btc_price_history)
                bollinger_bands = self._calc_bollinger_bands(self.btc_ohlc_history)
            else:
                rsi, stoch, atr, short_ma, long_ma, macd_results, bollinger_bands = [None]*7
                logger.warning("Tidak cukup data sejarah untuk mengira semua indikator untuk alert.")

            main_analysis = f"BTC: **${btc_price:,.2f}** ({btc_change:+.2f}%)\nETH: **${eth_price:,.2f}** ({eth_change:+.2f}%)"
            
            tech_analysis_list = []
            def is_valid_num(val): return val is not None and not math.isnan(val) and not math.isinf(val)
            
            if is_valid_num(rsi): tech_analysis_list.append(f"RSI({RSI_PERIOD}): `{rsi:.2f}` {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else ''}")
            if is_valid_num(stoch): tech_analysis_list.append(f"Stoch({STOCHASTIC_PERIOD}): `{stoch:.2f}` {'Overbought' if stoch > 80 else 'Oversold' if stoch < 20 else ''}")
            if is_valid_num(short_ma) and is_valid_num(long_ma):
                if short_ma > long_ma: tech_analysis_list.append("MA: Bullish Crossover")
                else: tech_analysis_list.append("MA: Bearish Crossover")
            if macd_results and is_valid_num(macd_results['macd_line']) and is_valid_num(macd_results['signal_line']):
                if macd_results['macd_line'] > macd_results['signal_line']: tech_analysis_list.append("MACD: Bullish Crossover")
                else: tech_analysis_list.append("MACD: Bearish Crossover")
            if bollinger_bands and all(is_valid_num(v) for v in bollinger_bands.values()):
                current_p = btc_price # Use current market price for current BB position
                if current_p is not None and bollinger_bands['upper'] is not None and bollinger_bands['lower'] is not None:
                    if current_p > bollinger_bands['upper']: tech_analysis_list.append("BB: Overbought (Near Upper Band)")
                    elif current_p < bollinger_bands['lower']: tech_analysis_list.append("BB: Oversold (Near Lower Band)")
                    else: tech_analysis_list.append("BB: Inside Bands")
            if is_valid_num(atr): tech_analysis_list.append(f"ATR({ATR_PERIOD}): `${atr:.2f}` (Volatiliti)")

            john_insight = self._generate_ai_insight(btc_change, rsi, stoch, macd_results, bollinger_bands, btc_price, "JohnAI")
            alpha_insight = self._generate_ai_insight(btc_change, rsi, stoch, macd_results, bollinger_bands, btc_price, "AlphaAI")
            
            fields = [
                ("Gambaran Pasaran", main_analysis, False),
                ("Analisis Teknikal Ringkas", "\n".join(tech_analysis_list) if tech_analysis_list else "Tidak cukup data untuk indikator.", False),
                ("Pandangan JohnAI", john_insight, False),
                ("Pandangan AlphaAI", alpha_insight, False),
                ("Risiko", self._risk_level(btc_change), True),
                ("Sentimen", self._get_sentiment(btc_change), True)
            ]
            
            embed_color = discord.Color.red() if alert_type == "Critical Shift" else discord.Color.orange() if alert_type == "Macro Shift" else discord.Color.blue()
            alert_embed = self._build_embed(
                f"🚨 Bitcoin {alert_type} Dikesan! 🚨",
                f"Pergerakan harga yang signifikan pada {self._now_utc()}.",
                fields, embed_color,
                footer_text="Ambil tindakan berdasarkan kajian anda sendiri. Ini bukan nasihat kewangan."
            )
            await self.alert_channel.send(embed=alert_embed)
            LAST_ALERT_TIME = current_time

        else:
            logger.info("Tiada pergerakan harga Bitcoin signifikan atau dalam cooldown untuk mencetuskan alert automatik.")
        
        self.last_btc_data['price'] = btc_price
        self.last_btc_data['change_24hr'] = btc_change
        self.last_eth_data['price'] = eth_price
        self.last_eth_data['change_24hr'] = eth_change
        

    # --- === EVENTS BOT === ---
    @commands.Cog.listener()
    async def on_ready(self):
        logger.info(f'{self.bot.user.name} berada dalam talian dan bersedia! ({self._now_utc()})')
        await self.bot.change_presence(activity=discord.Game(name=f"{BOT_PREFIX}btc | Sentinel V4.0 CMC"))
        
        self.alert_channel = self.bot.get_channel(ALERT_CHANNEL_ID)
        if not self.alert_channel:
            self._log_error(f"Saluran alert dengan ID {ALERT_CHANNEL_ID} tidak ditemui semasa startup.", "ChannelNotFound")

        # Initial data fetch using CMC for current price
        initial_current_data = await self._get_current_price_data()
        if initial_current_data and initial_current_data['btc'] is not None:
            self.last_btc_data['price'] = initial_current_data['btc']
            self.last_btc_data['change_24hr'] = initial_current_data['btc_change']
            self.last_eth_data['price'] = initial_current_data['eth']
            self.last_eth_data['change_24hr'] = initial_current_data['eth_change']
            
            # Fetch longer historical data for indicators. Use 'daily' interval for longer history.
            # Max days for free tier for daily is 90 days. We use 90 days as a base.
            # Ensure enough data for all indicator periods, plus a buffer.
            max_period_for_history = max(MA_LONG_PERIOD, BOLLINGER_PERIOD, MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD, RSI_PERIOD, STOCHASTIC_PERIOD, ATR_PERIOD)
            
            initial_historical = await self._get_historical_market_data('BTC', days=90, interval='daily') # Use daily for longer history
            
            if initial_historical and initial_historical['prices'] and len(initial_historical['prices']) >= max_period_for_history:
                self.btc_price_history.extend(initial_historical['prices'])
                self.btc_ohlc_history.extend(initial_historical['high_low_close'])
                logger.info(f"Sejarah harga BTC awal dimuatkan: {len(self.btc_price_history)} titik data.")
            else:
                self.btc_price_history.append(initial_current_data['btc'])
                # For initial point, create a dummy OHLC (as we don't have historical OHLC for single point)
                self.btc_ohlc_history.append((initial_current_data['btc'] * 1.005, initial_current_data['btc'] * 0.995, initial_current_data['btc']))
                logger.warning("Gagal memuatkan sejarah harga penuh dari CMC atau tidak cukup data. Menggunakan titik data semasa sebagai permulaan.")
            
            logger.info(f"Harga BTC awal ditetapkan: ${self.last_btc_data['price']:.2f}")
        else:
            logger.warning("Gagal mengambil data Bitcoin awal dari CMC. State harga akan menjadi 0.0 pada mulanya.")

        self.monitor_bitcoin_price.start()

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("Maaf, arahan itu tidak wujud. Cuba `!help` untuk senarai arahan.")
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"Anda terlepas hujah yang diperlukan: `{error.param.name}`. Sila semak `!help {ctx.command.name}`.")
        elif isinstance(error, commands.BadArgument):
             await ctx.send(f"Hujah tidak sah: `{error}`. Sila semak `!help {ctx.command.name}`.")
        else:
            self._log_error(error, f"CommandError_{ctx.command}")
            await ctx.send(f"Ralat berlaku semasa melaksanakan arahan. Sila cuba sebentar lagi. (Ralat: `{error}`)")

    # --- === ARAHAN BOT === ---

    @commands.command(name='btc', help='Dapatkan info BTC & ETH terkini dengan analisis AI.')
    async def get_current_status(self, ctx):
        logger.info(f"Arahan '{BOT_PREFIX}btc' dipanggil oleh {ctx.author} di {ctx.channel}.")
        await ctx.defer()

        current_data = await self._get_current_price_data()
        if not current_data or current_data['btc'] is None:
            await ctx.send("❌ Maaf, tidak dapat mengambil data Bitcoin sekarang. Sila cuba sebentar lagi.")
            return

        btc_price = current_data['btc']
        btc_change = current_data['btc_change']
        eth_price = current_data['eth']
        eth_change = current_data['eth_change']

        # Dapatkan indikator dari sejarah yang disimpan (jika cukup data)
        rsi, stoch, atr, short_ma, long_ma, macd_results, bollinger_bands = [None]*7
        
        min_data_needed_for_current_cmd = max(RSI_PERIOD, STOCHASTIC_PERIOD, ATR_PERIOD, MA_LONG_PERIOD, BOLLINGER_PERIOD, MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD)
        
        if len(self.btc_price_history) >= min_data_needed_for_current_cmd:
            rsi = self._calc_rsi(self.btc_price_history)
            stoch = self._calc_stochastic(self.btc_price_history)
            short_ma = self._calc_ma(self.btc_price_history, MA_SHORT_PERIOD)
            long_ma = self._calc_ma(self.btc_price_history, MA_LONG_PERIOD)
            macd_results = self._calc_macd(self.btc_price_history)
            bollinger_bands = self._calc_bollinger_bands(self.btc_ohlc_history)
            atr = self._calc_atr(self.btc_ohlc_history, ATR_PERIOD)

        macd_line = macd_results['macd_line'] if macd_results else None
        signal_line = macd_results['signal_line'] if macd_results else None
        histogram = macd_results['histogram'] if macd_results else None

        tech_analysis_summary = []
        def is_valid_num(val): return val is not None and not math.isnan(val) and not math.isinf(val)

        if is_valid_num(rsi): tech_analysis_summary.append(f"RSI({RSI_PERIOD}): `{rsi:.2f}`")
        if is_valid_num(stoch): tech_analysis_summary.append(f"Stoch({STOCHASTIC_PERIOD}): `{stoch:.2f}`")
        if is_valid_num(short_ma): tech_analysis_summary.append(f"MA({MA_SHORT_PERIOD}): `${short_ma:,.2f}`")
        if is_valid_num(long_ma): tech_analysis_summary.append(f"MA({MA_LONG_PERIOD}): `${long_ma:,.2f}`")
        if macd_line is not None and signal_line is not None: tech_analysis_summary.append(f"MACD: `{macd_line:.2f}` (Signal: `{signal_line:.2f}`)")
        if bollinger_bands and all(is_valid_num(v) for v in bollinger_bands.values()): tech_analysis_summary.append(f"BBands: (U: `${bollinger_bands['upper']:,.2f}` | M: `${bollinger_bands['middle']:,.2f}` | L: `${bollinger_bands['lower']:,.2f}`)")
        if is_valid_num(atr): tech_analysis_summary.append(f"ATR({ATR_PERIOD}): `${atr:.2f}` (Volatiliti)")


        john_insight = self._generate_ai_insight(btc_change, rsi, stoch, macd_results, bollinger_bands, btc_price, "JohnAI")
        alpha_insight = self._generate_ai_insight(btc_change, rsi, stoch, macd_results, bollinger_bands, btc_price, "AlphaAI")

        fields = [
            ("Harga Semasa", 
             f"BTC: **${btc_price:,.2f}** ({btc_change:+.2f}% {self._get_trend_icon(btc_change)})\n"
             f"ETH: **${eth_price:,.2f}** ({eth_change:+.2f}% {self._get_trend_icon(eth_change)})",
             False),
            ("Indikator Utama", "\n".join(tech_analysis_summary) if tech_analysis_summary else "Tidak cukup data untuk indikator.", False),
            ("Pandangan JohnAI (Trend Analyst)", john_insight, False),
            ("Pandangan AlphaAI (Risk & Volatility Analyst)", alpha_insight, False),
            ("Tahap Risiko", self._risk_level(btc_change), True),
            ("Sentimen Pasaran", self._get_sentiment(btc_change), True)
        ]

        embed = self._build_embed(
            f"📊 Laporan Pasaran Kripto ({self._now_utc()})",
            f"Analisis pantas Bitcoin & Ethereum:",
            fields, discord.Color.dark_purple()
        )
        await ctx.send(embed=embed)

    @commands.command(name='report', help='Dapatkan laporan analisis teknikal mendalam (1D, 7D, 30D, 90D). Contoh: !report btc 7D')
    async def get_detailed_report(self, ctx, coin_id: str = 'btc', days_str: str = '7D'):
        logger.info(f"Arahan '{BOT_PREFIX}report {coin_id} {days_str}' dipanggil oleh {ctx.author} di {ctx.channel}.")
        await ctx.defer()

        coin_map = {'btc': 'BTC', 'eth': 'ETH'} # Use CMC symbols
        
        # Decide interval based on days. CMC free tier hourly is limited to 7 days.
        # For anything beyond 7 days, we use daily.
        days_map = {'1D': {'days': 1, 'interval': 'hourly'},
                    '7D': {'days': 7, 'interval': 'hourly'},
                    '30D': {'days': 30, 'interval': 'daily'}, # Beyond 7 days, use daily
                    '90D': {'days': 90, 'interval': 'daily'}}

        coin_symbol = coin_map.get(coin_id.lower())
        selected_period_info = days_map.get(days_str.upper())

        if not coin_symbol:
            await ctx.send("❌ ID koin tidak sah. Sila gunakan `btc` atau `eth`.")
            return
        if not selected_period_info:
            await ctx.send("❌ Jangka masa laporan tidak sah. Sila gunakan `1D`, `7D`, `30D`, atau `90D`.")
            return

        days_to_fetch = selected_period_info['days']
        interval_to_fetch = selected_period_info['interval']

        report_embed = await self._perform_detailed_analysis(coin_symbol, days_to_fetch, interval_to_fetch)
        if isinstance(report_embed, str):
            await ctx.send(report_embed)
        else:
            await ctx.send(embed=report_embed)

    @commands.command(name='help', help='Tunjukkan senarai arahan bot ini.')
    async def help_command(self, ctx):
        embed = discord.Embed(
            title="Panduan Bot Bitcoin Sentinel AI (Advance V4.0 - CMC)",
            description="Berikut adalah senarai arahan yang boleh anda gunakan:",
            color=discord.Color.blue()
        )
        embed.add_field(name="`!btc`", value="Dapatkan gambaran pasaran Bitcoin & Ethereum terkini, termasuk analisis AI dan indikator utama (dikemaskini setiap jam).", inline=False)
        embed.add_field(name="`!report <koin_id> <jangka_masa>`", value="Dapatkan laporan analisis teknikal mendalam untuk Bitcoin (`btc`) atau Ethereum (`eth`) dalam jangka masa tertentu (`1D` (hourly), `7D` (hourly), `30D` (daily), `90D` (daily)). Contoh: `!report btc 7D`", inline=False)
        embed.add_field(name="`!help`", value="Tunjukkan panduan ini.", inline=False)
        embed.set_footer(text="Fokus pada keputusan yang berinformasi. Ini bukan nasihat kewangan.")
        await ctx.send(embed=embed)


# --- === JALANKAN BOT === ---
if __name__ == '__main__':
    # Initial check for all required environment variables
    missing_vars = []
    if not DISCORD_TOKEN:
        missing_vars.append("DISCORD_BOT_TOKEN")
    if not CMC_API_KEY:
        missing_vars.append("CMC_API_KEY")
    
    try:
        if not os.getenv("BITCOIN_ALERT_CHANNEL_ID"): # Check if it's set at all before int conversion
            missing_vars.append("BITCOIN_ALERT_CHANNEL_ID")
        else:
            ALERT_CHANNEL_ID = int(os.getenv("BITCOIN_ALERT_CHANNEL_ID"))
            if ALERT_CHANNEL_ID == 0:
                raise ValueError("BITCOIN_ALERT_CHANNEL_ID tidak sah (mesti nombor > 0).")
    except ValueError as e:
        logger.critical(f"Ralat Konfigurasi: {e}. Keluar.")
        print(f"\n--- RALAT PENTING ---\nRalat Konfigurasi: {e}\nSila pastikan 'BITCOIN_ALERT_CHANNEL_ID' adalah ID saluran Discord yang sah (hanya nombor).\n--------------------\n")
        exit()


    if missing_vars:
        logger.critical(f"Pemboleh ubah persekitaran yang diperlukan tidak ditetapkan: {', '.join(missing_vars)}. Keluar.")
        print("\n--- RALAT PENTING ---")
        print(f"Sila tetapkan pemboleh ubah persekitaran berikut: {', '.join(missing_vars)}.")
        print("Contoh (Linux/macOS):")
        print("export DISCORD_BOT_TOKEN='TOKEN_ANDA_DI_SINI'")
        print("export CMC_API_KEY='KUNCI_API_CMC_ANDA_DI_SINI'")
        print("export BITCOIN_ALERT_CHANNEL_ID='ID_SALURAN_ANDA_DI_SINI'")
        print("\nContoh (Windows Command Prompt):")
        print("set DISCORD_BOT_TOKEN='TOKEN_ANDA_DI_SINI'")
        print("set CMC_API_KEY='KUNCI_API_CMC_ANDA_DI_SINI'")
        print("set BITCOIN_ALERT_CHANNEL_ID='ID_SALURAN_ANDA_DI_SINI'")
        print("--------------------\n")
        exit()


    try:
        logger.info("Mencuba untuk menjalankan bot Discord...")
        intents = discord.Intents.default()
        intents.message_content = True 
        bot = commands.Bot(command_prefix=BOT_PREFIX, intents=intents)
        asyncio.run(bot.add_cog(BitcoinSentinelAdvance(bot)))
        bot.run(DISCORD_TOKEN)
    except discord.LoginFailure:
        logger.critical("Token Discord yang tidak sah. Sila semak DISCORD_BOT_TOKEN anda.")
    except Exception as e:
        logger.critical(f"Ralat tidak dijangka berlaku semasa permulaan bot: {e}", exc_info=True)

