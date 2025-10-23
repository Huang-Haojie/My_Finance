import os
from pathlib import Path
import gradio as gr
from stock import Stock, is_valid_ticker, get_risk_free_rate
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy.stats import norm
from functions import (
    calculate_covariance_matrix,
    optimize_minimum_variance,
    optimize_maximum_sharpe,
    calculate_portfolio_std,
    calculate_portfolio_sharpe,
    bsm_pricer,
    bsm_ivol,
    bsm_greeks
)

class StockWebUI:
    def __init__(self):
        self.current_stock = None

    def watch_list(self,ticker=None):
        """
        add ticker to watchlist if provided
        return watchlist as list, new Dropdown box, and str for status
        """
        if not os.path.exists('./watchlist'):
            path = Path('./watchlist')
            path.touch()
            with open('./watchlist','w') as f:
                f.write('')

        with open('watchlist', 'r') as f:
            watchlist = f.readlines()
        for i in range(len(watchlist)):
            watchlist[i] = watchlist[i].strip()

        if ticker is not None:
            if is_valid_ticker(ticker):
                if ticker not in watchlist:
                    with open('watchlist', 'a') as f:
                        watchlist.append(ticker)
                        f.write(ticker + '\n')
                    return gr.Dropdown(choices=watchlist), 'Ticker Added to Watchlist'
                else:
                    return gr.Dropdown(choices=watchlist), 'Ticker Already Exists'
            else:
                return gr.Dropdown(choices=watchlist), 'Invalid Ticker'
        else:
            return watchlist

    def remove_watch_list(self,ticker):
        "remove given ticker from watchlist and return new dropdown box, status message"
        if not os.path.exists('./watchlist'):
            path = Path('./watchlist')
            path.touch()
            with open('./watchlist','w') as f:
                f.write('')

        with open('watchlist', 'r') as f:
            watchlist = f.readlines()
        for i in range(len(watchlist)):
            watchlist[i] = watchlist[i].strip()

        if ticker in watchlist:
            watchlist.remove(ticker)
            for i in range(len(watchlist)):
                watchlist[i] = watchlist[i]+'\n'
            with open('watchlist', 'w') as f:
                f.writelines(watchlist)
                return gr.Dropdown(choices=watchlist), 'Ticker Removed from Watchlist'
        else:
            return gr.Dropdown(choices=watchlist), 'Given Ticker Not Found'

    def get_stock_info(self, ticker, period='1y'):
        """Fetch and display basic stock information"""
        if not ticker or ticker.strip() == "":
            return None, None, "Please enter a valid ticker symbol"

        ticker = ticker.strip().upper()

        try:
            # Create stock instance
            self.current_stock = Stock(ticker)

            # Get price history for chart first to validate ticker
            price_history = self.current_stock.get_price_history(period=period)

            if price_history.empty:
                return None, None, f"No price data found for {ticker}. Please check the ticker symbol."

            # Fetch basic information
            current_price = self.current_stock.get_current_price()
            hist_vol = self.current_stock.get_historical_volatility()
            dividend_yield = self.current_stock.get_dividend_yield()
            risk_free_rate = self.current_stock.get_risk_free_rate()

            # Get stock info from yfinance
            info = self.current_stock.prices.info

            # Prepare information dictionary
            stock_data = {
                'Metric': [
                    'Company Name',
                    'Ticker',
                    'Current Price',
                    'Market Cap',
                    'PE Ratio',
                    '52 Week High',
                    '52 Week Low',
                    'Historical Volatility (1Y)',
                    'Dividend Yield',
                    'Risk-Free Rate',
                    'Sector',
                    'Industry'
                ],
                'Value': [
                    info.get('longName', 'N/A'),
                    ticker,
                    f"${current_price:.2f}" if current_price else 'N/A',
                    f"${info.get('marketCap', 0):,.0f}" if info.get('marketCap') else 'N/A',
                    f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A',
                    f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                    f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A',
                    f"{hist_vol:.2%}" if hist_vol else 'N/A',
                    f"{dividend_yield:.2%}" if dividend_yield else '0.00%',
                    f"{risk_free_rate:.2%}",
                    info.get('sector', 'N/A'),
                    info.get('industry', 'N/A')
                ]
            }

            info_df = pd.DataFrame(stock_data)

            # Prepare chart data
            chart_data = price_history.reset_index()[['Date', 'Close']].rename(columns={'Close': 'Price'})

            return info_df, chart_data, f"Successfully loaded data for {ticker}"

        except Exception as e:
            import traceback
            error_msg = f"Error fetching data for {ticker}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # For debugging
            return None, None, f"Error fetching data for {ticker}: {str(e)}"

    def portfolio_refresh(self, current_df):
        """Refresh portfolio composition based on current watchlist (equal weight)"""
        tickers = self.watch_list()
        if len(tickers) == 0:
            return pd.DataFrame({"Ticker": [], "Weight": []})

        new_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight": [1.0/len(tickers)] * len(tickers)
        })
        return new_df, 'Portfolio weights refreshed'

    def portfolio_regularize(self, portfolio_df):
        """Normalize portfolio weights to sum to 1"""
        if portfolio_df is None or len(portfolio_df) == 0:
            return portfolio_df

        weights = np.array(portfolio_df['Weight'].values, dtype=float)
        normalized_weights = np.ones(len(weights))/len(weights)

        result_df = portfolio_df.copy()
        result_df['Weight'] = normalized_weights
        return result_df, 'Portfolio weights regularized'

    def portfolio_optimize_risk(self, portfolio_df, period='1y'):
        """Optimize portfolio weights for minimum risk (variance)"""
        if portfolio_df is None or len(portfolio_df) == 0:
            return portfolio_df, 'Invalid portfolio data'

        tickers = portfolio_df['Ticker'].tolist()

        try:
            # Fetch price data for all tickers
            price_data = pd.DataFrame()
            for ticker in tickers:
                stock_obj = Stock(ticker)
                hist = stock_obj.get_price_history(period=period)
                if not hist.empty:
                    price_data[ticker] = hist['Close']

            if price_data.empty or len(price_data.columns) < 2:
                return portfolio_df, 'Invalid portfolio data'

            # Calculation
            returns = price_data.pct_change().dropna()
            optimal_weights = optimize_minimum_variance(returns)

            result_df = portfolio_df.copy()
            result_df['Weight'] = optimal_weights
            return result_df, 'Portfolio weights optimized for minimum risk'

        except Exception as e:
            print(f"Error optimizing portfolio: {e}")
            return portfolio_df

    def portfolio_optimize_beta(self, portfolio_df, period='1y'):
        """Optimize portfolio weights for maximum Sharpe ratio (return/risk)"""
        if portfolio_df is None or len(portfolio_df) == 0:
            return portfolio_df,'Invalid portfolio data'

        tickers = portfolio_df['Ticker'].tolist()

        try:
            # Fetch price data for all tickers
            price_data = pd.DataFrame()
            for ticker in tickers:
                stock_obj = Stock(ticker)
                hist = stock_obj.get_price_history(period=period)
                if not hist.empty:
                    price_data[ticker] = hist['Close']

            if price_data.empty or len(price_data.columns) < 2:
                return portfolio_df, 'Invalid portfolio data'
            rf_rate = get_risk_free_rate()
            # Calculate returns and covariance
            returns = price_data.pct_change().dropna()

            # Optimize for maximum Sharpe ratio
            optimal_weights = optimize_maximum_sharpe(
                returns,
                rf_rate,
            )

            result_df = portfolio_df.copy()
            result_df['Weight'] = optimal_weights
            return result_df, 'Portfolio weights optimized for maximum Sharpe ratio'

        except Exception as e:
            print(f"Error optimizing portfolio: {e}")
            return portfolio_df

    def portfolio_calculate_metrics(self, portfolio_df, period='1y'):
        """Calculate portfolio metrics"""
        if portfolio_df is None or len(portfolio_df) == 0:
            return None, None, 'Invalid portfolio data'

        tickers = portfolio_df['Ticker'].tolist()
        weights = np.array(portfolio_df['Weight'].values, dtype=float)

        try:
            # Fetch price data for all tickers
            price_data = pd.DataFrame()
            for ticker in tickers:
                stock_obj = Stock(ticker)
                hist = stock_obj.get_price_history(period=period)
                if not hist.empty:
                    price_data[ticker] = hist['Close']

            if price_data.empty:
                return None, None, 'Historical price not found'

            # Calculate returns
            returns = price_data.pct_change().dropna()

            # Calculate annualization factor based on actual number of data points and period
            num_periods = len(returns)
            if num_periods > 0:
                annualization_factor = 252 / num_periods
            else:
                annualization_factor = 1

            annualized_returns = (returns + 1).prod() ** annualization_factor - 1

            cov_matrix = calculate_covariance_matrix(price_data)

            # Calculate portfolio metrics
            portfolio_return = np.dot(weights, annualized_returns.values)
            portfolio_std = calculate_portfolio_std(weights, cov_matrix.values)
            rf_rate = get_risk_free_rate()
            portfolio_sharpe = calculate_portfolio_sharpe(weights, returns,rf_rate)

            # Calculate portfolio value over time
            np_weights = np.array(weights*100)

            portfolio_value = (np_weights * (1+returns).cumprod()).sum(axis=1)



            # Prepare metrics dataframe
            metrics_data = {
                'Metric': [
                    'Expected Annual Return',
                    'Annual Volatility (Risk)',
                    'Sharpe Ratio',
                    'Number of Assets',
                    'Total Weight'
                ],
                'Value': [
                    f"{portfolio_return:.2%}",
                    f"{portfolio_std:.2%}",
                    f"{portfolio_sharpe:.4f}",
                    f"{len(tickers)}",
                    f"{weights.sum():.4f}"
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)

            # Prepare chart data
            chart_data = pd.DataFrame({
                'Date': portfolio_value.index,
                'Value': portfolio_value.values
            })

            return metrics_df, chart_data, 'Portfolio metrics calculated'

        except Exception as e:
            import traceback
            print(f"Error calculating metrics: {e}")
            print(traceback.format_exc())
            return None, None

    def get_option_expiry_dates(self, ticker):
        """Get available option expiration dates for a ticker"""
        if not ticker or ticker.strip() == "":
            return gr.Dropdown(choices=[]), "Please enter a valid ticker symbol"

        ticker = ticker.strip().upper()

        try:
            stock = Stock(ticker)
            expiry_dates = stock.get_option_expiry_dates()

            if not expiry_dates:
                return gr.Dropdown(choices=[]), f"No option data available for {ticker}"

            return gr.Dropdown(choices=expiry_dates), f"Found {len(expiry_dates)} expiration dates for {ticker}"

        except Exception as e:
            return gr.Dropdown(choices=[]), f"Error fetching expiry dates: {str(e)}"

    def calculate_option_metrics(self, ticker, expiry_date):
        """Calculate option Greeks and fair prices for all options in the chain"""
        if not ticker or ticker.strip() == "":
            return None, None, None, "Please enter a valid ticker symbol"

        if not expiry_date:
            return None, None, None, "Please select an expiration date"

        ticker = ticker.strip().upper()

        try:
            # Create stock instance
            stock = Stock(ticker)

            # Get current stock price and parameters
            S = stock.get_current_price()
            if S is None:
                return None, None, None, f"Could not fetch current price for {ticker}"

            r = stock.get_risk_free_rate()
            y = stock.get_dividend_yield()

            # Get option chain
            calls, puts = stock.get_option_chain(expiry_date)

            if calls is None or puts is None:
                return None, None, None, f"Could not fetch option chain for {ticker} expiring {expiry_date}"

            # Calculate time to expiration in years
            expiry_datetime = datetime.strptime(expiry_date, '%Y-%m-%d')
            today = datetime.now()
            T = (expiry_datetime - today).days / 365.0

            if T <= 0:
                return None, None, None, "Expiration date must be in the future"

            # Process calls
            call_data = []
            iv_data_calls = []

            for _, option in calls.iterrows():
                K = option['strike']
                market_price = option['lastPrice']
                sigma = Stock(ticker).get_historical_volatility()
                try:
                    # Calculate theoretical price

                    theoretical_price = bsm_pricer(S, K, r, y, sigma, T, 'c')  # Using 0.2 as initial sigma

                    # Calculate implied volatility
                    if market_price > 0:  # Only calculate IV for options with meaningful prices

                        iv = bsm_ivol(S, K, r, y, T, 'c', market_price, 0.8)

                        # Calculate Greeks using implied volatility
                        gamma, vega, theta, rho = bsm_greeks(S, K, r, iv, T, 'c')

                        # Delta calculation
                        d1 = (np.log(S / K) + (r - y + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
                        delta = np.exp(-y * T) * norm.cdf(d1)

                        # Recalculate theoretical price with IV
                        #theoretical_price = bsm_pricer(S, K, r, y, iv, T, 'c')
                        #print(theoretical_price)

                        iv_data_calls.append({'Strike': K, 'IV': iv})
                    else:
                        delta = gamma = vega = theta = rho = iv = None

                    call_data.append({
                        'Strike': K,
                        'Market Price': f"${market_price:.2f}",
                        'Theoretical Price': f"${theoretical_price:.2f}" if theoretical_price else 'N/A',
                        'Implied Vol': f"{iv:.2%}" if iv else 'N/A',
                        'Delta': f"{delta:.4f}" if delta else 'N/A',
                        'Gamma': f"{gamma:.4f}" if gamma else 'N/A',
                        'Vega': f"{vega:.4f}" if vega else 'N/A',
                        'Theta': f"{theta:.4f}" if theta else 'N/A',
                        'Rho': f"{rho:.4f}" if rho else 'N/A',
                        'Volume': option.get('volume', 0),
                        'Open Interest': option.get('openInterest', 0)
                    })
                except Exception as e:
                    print(f"Error processing call option at strike {K}: {e}")
                    continue

            # Process puts
            put_data = []
            iv_data_puts = []

            for _, option in puts.iterrows():
                K = option['strike']
                market_price = option['lastPrice']

                try:
                    # Calculate theoretical price
                    theoretical_price = bsm_pricer(S, K, r, y, sigma, T, 'p')

                    # Calculate implied volatility
                    if market_price > 0:
                        iv = bsm_ivol(S, K, r, y, T, 'p', market_price, 0.8)

                        # Calculate Greeks
                        gamma, vega, theta, rho = bsm_greeks(S, K, r, iv, T, 'p')

                        # Delta calculation for put
                        d1 = (np.log(S / K) + (r - y + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
                        delta = -np.exp(-y * T) * norm.cdf(-d1)

                        # Recalculate theoretical price with IV
                        #theoretical_price = bsm_pricer(S, K, r, y, iv, T, 'p')

                        iv_data_puts.append({'Strike': K, 'IV': iv})
                    else:
                        delta = gamma = vega = theta = rho = iv = None

                    put_data.append({
                        'Strike': K,
                        'Market Price': f"${market_price:.2f}",
                        'Theoretical Price': f"${theoretical_price:.2f}" if theoretical_price else 'N/A',
                        'Implied Vol': f"{iv:.2%}" if iv else 'N/A',
                        'Delta': f"{delta:.4f}" if delta else 'N/A',
                        'Gamma': f"{gamma:.4f}" if gamma else 'N/A',
                        'Vega': f"{vega:.4f}" if vega else 'N/A',
                        'Theta': f"{theta:.4f}" if theta else 'N/A',
                        'Rho': f"{rho:.4f}" if rho else 'N/A',
                        'Volume': option.get('volume', 0),
                        'Open Interest': option.get('openInterest', 0)
                    })
                except Exception as e:
                    print(f"Error processing put option at strike {K}: {e}")
                    continue

            # Create DataFrames
            call_df = pd.DataFrame(call_data)
            put_df = pd.DataFrame(put_data)

            # Create volatility smile chart data
            iv_chart_data = []
            for item in iv_data_calls:
                iv_chart_data.append({
                    'Strike': item['Strike'],
                    'Implied Volatility': item['IV'],
                    'Type': 'Call'
                })
            for item in iv_data_puts:
                iv_chart_data.append({
                    'Strike': item['Strike'],
                    'Implied Volatility': item['IV'],
                    'Type': 'Put'
                })

            iv_chart_df = pd.DataFrame(iv_chart_data)

            status_msg = f"Successfully calculated metrics for {ticker} options expiring {expiry_date}\nCurrent Stock Price: ${S:.2f}, Time to Expiry: {T:.2f} years"

            return call_df, put_df, iv_chart_df, status_msg

        except Exception as e:
            import traceback
            error_msg = f"Error calculating option metrics: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, None, None, f"Error: {str(e)}"

    def create_interface(self):
        """Create and return the Gradio interface"""
        with gr.Blocks(title="Stock Information Dashboard") as demo:
            gr.Markdown("# Stock Information Dashboard")
            gr.Markdown("Enter a stock ticker symbol to view basic information and price history")

            with gr.Row():
                """ticker_input = gr.Textbox(
                    label="Stock Ticker",
                    placeholder="e.g., AAPL, MSFT, GOOGL",
                    scale=3
                )"""

                ticker_input = gr.Dropdown(
                    choices=self.watch_list(),
                    allow_custom_value=True,  # Key parameter!
                    label="Enter Stock Ticker or Select From Watchlist",
                    #placeholder="e.g., AAPL, MSFT, GOOGL"
                    scale=3
                )
                with gr.Column():
                    like_btn = gr.Button("Add To Watch List", variant="primary", scale=1)
                    del_btn = gr.Button("Remove", variant="primary", scale=1)

            status_output = gr.Textbox(label="Status", interactive=False)

            with gr.Tab("Basic Info"):
                with gr.Row():
                    stock_period = gr.Dropdown(
                        choices=['5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'],
                        value='1y',
                        label="Time Period",
                        scale=3
                    )
                    submit_btn = gr.Button("Get Stock Info", variant="primary", scale=1)

                with gr.Row():
                    info_output = gr.Dataframe(
                        label="Stock Information",
                        headers=['Metric', 'Value'],
                        interactive=False
                    )

                with gr.Row():
                    price_chart = gr.LinePlot(
                        x='Date',
                        y='Price',
                        title='Price History',
                        width=800,
                        height=400
                    )

                # Set up event handlers
                submit_btn.click(
                    fn=self.get_stock_info,
                    inputs=[ticker_input, stock_period],
                    outputs=[info_output, price_chart, status_output]
                )

                like_btn.click(
                    fn=self.watch_list,
                    inputs=[ticker_input],
                    outputs=[ticker_input, status_output]
                )

                del_btn.click(
                    fn=self.remove_watch_list,
                    inputs=[ticker_input],
                    outputs=[ticker_input, status_output]
                )

            with gr.Tab("Option Chain"):
                with gr.Row():
                    option_expiry = gr.Dropdown(
                        choices=[],
                        label="Select Expiration Date",
                        scale=2
                    )
                    with gr.Column():
                        load_expiry_btn = gr.Button("Load Expiry Dates", variant="secondary", scale=1)
                        calculate_greeks_btn = gr.Button("Calculate Metrics", variant="primary", scale=1)



                with gr.Row():
                    call_options_output = gr.Dataframe(
                        label="Call Options Chain",
                        interactive=False,
                        headers=['Strike','Market Price','Theoretical Price','Implied Vol','Delta','Gamma','Vega','Theta','Rho','Volume','Open Interest'],
                    )

                with gr.Row():
                    put_options_output = gr.Dataframe(
                        label="Put Options Chain",
                        interactive=False,
                        headers=['Strike','Market Price','Theoretical Price','Implied Vol','Delta','Gamma','Vega','Theta','Rho','Volume','Open Interest'],
                    )

                with gr.Row():
                    volatility_smile_chart = gr.LinePlot(
                        x='Strike',
                        y='Implied Volatility',
                        color='Type',
                        title='Volatility Smile (Implied Volatility vs Strike Price)',
                        width=800,
                        height=400
                    )

                # Connect option chain handlers
                load_expiry_btn.click(
                    fn=self.get_option_expiry_dates,
                    inputs=[ticker_input],
                    outputs=[option_expiry, status_output]
                )

                calculate_greeks_btn.click(
                    fn=self.calculate_option_metrics,
                    inputs=[ticker_input, option_expiry],
                    outputs=[call_options_output, put_options_output, volatility_smile_chart, status_output]
                )

            with gr.Tab("Portfolio"):

                with gr.Row():
                    portfolio_period = gr.Dropdown(
                        choices=['5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'],
                        value='1y',
                        label="Time Period for Analysis",
                        scale=2
                    )

                with gr.Row():
                    tickers = self.watch_list()
                    port_comp_init = pd.DataFrame({
                        "Ticker": tickers,
                        "Weight": [1/len(tickers)] * len(tickers) if len(tickers) > 0 else [],
                    })
                    port_comp = gr.Dataframe(
                        label="Portfolio Composition",
                        headers=['Ticker', 'Weight'],
                        interactive=True,
                        value=port_comp_init,
                        scale = 3
                    )

                    with gr.Column():
                        refresh = gr.Button("Refresh", variant="primary", scale=1)  #deal with updates on watchlist
                        regularization = gr.Button("Regularization", variant="primary", scale=1)  #ensure the given weights sum to 1
                        optimize_risk = gr.Button("Optimize Risk", variant="primary", scale=1)  #lowest risk
                        optimize_beta = gr.Button("Optimize Beta(Return/Risk)", variant="primary", scale=1) #highest return on risk
                        cal_matrics = gr.Button("Calculate Matrics", variant="primary", scale=1)

                with gr.Row():
                    port_info_output = gr.Dataframe(
                        label="Portfolio Information",
                        headers=['Metric', 'Value'],
                        interactive=False
                    )

                with gr.Row():
                    port_price_chart = gr.LinePlot(
                        x='Date',
                        y='Value',
                        title='Portfolio Value History',
                        width=800,
                        height=400
                    )

                # Connect portfolio button handlers
                refresh.click(
                    fn=self.portfolio_refresh,
                    inputs=[port_comp],
                    outputs=[port_comp,status_output]
                )

                regularization.click(
                    fn=self.portfolio_regularize,
                    inputs=[port_comp],
                    outputs=[port_comp,status_output]
                )

                optimize_risk.click(
                    fn=self.portfolio_optimize_risk,
                    inputs=[port_comp, portfolio_period],
                    outputs=[port_comp,status_output]
                )

                optimize_beta.click(
                    fn=self.portfolio_optimize_beta,
                    inputs=[port_comp, portfolio_period],
                    outputs=[port_comp,status_output]
                )

                cal_matrics.click(
                    fn=self.portfolio_calculate_metrics,
                    inputs=[port_comp, portfolio_period],
                    outputs=[port_info_output, port_price_chart, status_output]
                )

            return demo

    def launch(self, **kwargs):
        """Launch the web interface"""
        demo = self.create_interface()
        demo.launch(**kwargs)


# For direct execution
if __name__ == "__main__":
    ui = StockWebUI()
    ui.launch()