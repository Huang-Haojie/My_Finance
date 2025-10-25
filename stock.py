import yfinance as yf
import numpy as np

class Stock:
    def __init__(self, stock_ticker, risk_free_rate=0.05):
        self.ticker = stock_ticker
        self.prices = yf.Ticker(stock_ticker)
        self.risk_free_rate = risk_free_rate

    def get_current_price(self):
        """Get the current stock price"""
        try:
            return self.prices.info.get('currentPrice') or self.prices.info.get('regularMarketPrice')
        except:
            hist = self.prices.history(period='1d')
            return hist['Close'].iloc[-1] if not hist.empty else None

    def get_price_history(self, period='1y', interval='1d'):
        """Get historical price data"""
        return self.prices.history(period=period, interval=interval)

    def get_historical_volatility(self, period='1y', trading_days=252):
        """Calculate historical volatility (annualized )"""
        hist = self.get_price_history(period=period)
        if hist.empty or len(hist) < 2:
            return None

        log_returns = np.log(hist['Close'] / hist['Close'].shift(1))
        return log_returns.std() * np.sqrt(trading_days)

    def get_dividend_yield(self):
        """Get the current dividend yield"""
        try:
            div_yield = self.prices.info.get('dividendYield')
            return div_yield if div_yield else 0.0
        except:
            return 0.0

    def get_risk_free_rate(self):
        """Get the risk-free rate (can be overridden in constructor)"""
        self.risk_free_rate=get_risk_free_rate()
        return self.risk_free_rate

    def get_option_expiry_dates(self):
        """Get available option expiration dates"""
        try:
            return self.prices.options
        except:
            return []

    def get_option_chain(self, expiry_date):
        """Get option chain for a specific expiration date"""
        try:
            opt = self.prices.option_chain(expiry_date)
            return opt.calls, opt.puts
        except Exception as e:
            print(f"Error fetching option chain: {e}")
            return None, None


def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'regularMarketPrice' not in info:
            return False

        return True
    except:
        return False

def get_risk_free_rate():
    treasury = yf.Ticker("^TNX")
    current_yield = treasury.info.get('regularMarketPrice')
    return current_yield/100

def get_sofr_30d():
    """Get the 30-day SOFR (Secured Overnight Financing Rate) as the risk-free rate."""
    try:
        treasury_4week = yf.Ticker("^IRX")
        rate = treasury_4week.info.get('regularMarketPrice')
        return rate / 100

    except Exception as e:
        return get_risk_free_rate()

if __name__ == "__main__":
    print(get_sofr_30d())
    print(get_risk_free_rate())