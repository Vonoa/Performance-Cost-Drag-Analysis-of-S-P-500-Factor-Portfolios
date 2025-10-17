import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from io import StringIO

#Scraping the current list of S&P 500 tickets from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
html = response.text
sp500_table = pd.read_html(StringIO(html))[0]

#Extract tickers and clean up
tickers = sp500_table['Symbol'].tolist()
tickers = [t.replace('.', '-') for t in tickers]

#testing period
start_date = "2018-01-01"
end_date = "2025-10-14"  

#download adjusted close price data histroy for all tickers
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

#yfinance returns a MultiIndex converts back into simple columns
if isinstance(data.columns, pd.MultiIndex): 
    data = data["Adj Close"]  

#Dsicards tickers that are missing more then 50% of data
data = data.dropna(axis=1, thresh=int(0.5 * len(data)))
#Daily precentage returns
returns = data.pct_change(fill_method=None)

#Simulates an equally weighted portfolio with quarterly rebalancing
def equal_weight_portfolio(returns_data, rebalance_days=63): 
    #Data frame to store daily weights
    weights = pd.DataFrame(index=returns_data.index, columns=returns_data.columns, data=0.0)
    #Series to hold the current day's weights
    current_weights = pd.Series(0.0, index=returns_data.columns)

    for i, date in enumerate(returns_data.index):
        #rebalance every quarter
        if i % rebalance_days == 0: 
            if i > 0:
                available_tickers = returns_data.iloc[i-1].dropna().index.tolist()
            else:
                # Use current day for the very first initialization
                available_tickers = returns_data.iloc[i].dropna().index.tolist()
            
            if available_tickers:
                n = len(available_tickers)
                current_weights[:] = 0.0 #Resets all weight to 0
                current_weights.loc[available_tickers] = 1.0 / n #Set equal weights
        #Store the weights of current day
        weights.loc[date] = current_weights
    #Drop the first day
    returns_data = returns_data.iloc[1:]
    weights = weights.iloc[1:]
    #Portfolio returns
    daily_port_returns = (returns_data.fillna(0) * weights).sum(axis=1)
    return daily_port_returns.dropna()
#Raw Equal-Weights
eq_returns = equal_weight_portfolio(returns)
#Define Assumptions
annual_expense_ratio = 0.0020
annual_transaction_cost = 0.0035
annual_drag = annual_expense_ratio + annual_transaction_cost
#Converts the annual costs to a daily cost
daily_expense_eq_realisitic = annual_drag / 252
daily_expense_eq_no_drag = annual_expense_ratio / 252
#Creates two simulated portfolios
eq_returns_realistic = eq_returns - daily_expense_eq_realisitic
eq_cum_realisitc = (1 + eq_returns_realistic).cumprod()

eq_returns_no_drag = eq_returns - daily_expense_eq_no_drag
eq_cum_no_drag = (1 + eq_returns_no_drag).cumprod()

# Download benchmark data (SPY and RSP)
rsp = yf.download("RSP", start=start_date, end=end_date, auto_adjust=True)
spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True)
#Calcuate cumalative returns for benchmarks
spy_cum = (1 + spy["Close"].pct_change().dropna()).cumprod()
rsp_cum = (1 + rsp["Close"].pct_change().dropna()).cumprod()
# Align the dates
common_dates = eq_cum_realisitc.index.intersection(spy_cum.index).intersection(rsp_cum.index)
#Align all cumulative returns to common range
eq_cum_aligned_realisitc = eq_cum_realisitc.loc[common_dates]
eq_cum_aligned_no_drag = eq_cum_no_drag.loc[common_dates]
spy_cum_aligned = spy_cum.loc[common_dates]
rsp_cum_aligned = rsp_cum.loc[common_dates]
#Helper Function due to Pandas
def to_scalar(value):
    if isinstance(value, pd.Series):
        return value.item()
    return value

#Performance metrics function
def performance_metrics(daily_returns, cumulative):
    
    daily_returns = daily_returns.dropna()
    cumulative = cumulative.dropna()
    
    total_return = to_scalar(cumulative.iloc[-1] - 1)
    if len(daily_returns) < 252:
        annualized_return = np.nan
        annualized_vol = np.nan
    else:
        annualized_return = to_scalar((1 + total_return) ** (252 / len(daily_returns)) - 1)
        annualized_vol = to_scalar(daily_returns.std() * np.sqrt(252))
    if annualized_vol == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = annualized_return / annualized_vol
    
    # Calculate Max Drawdown
    max_drawdown = to_scalar((cumulative / cumulative.cummax() - 1).min())
    return {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }
#Calcuate Drawdown for Second Plot
def calculate_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.dropna()
#Calcuate returns for Benchmarks
spy_returns = spy["Close"].pct_change().dropna()
rsp_returns = rsp["Close"].pct_change().dropna()
#Calcuate metrics for all four portfolios
eq_metrics_realisitc = performance_metrics(eq_returns_realistic, eq_cum_realisitc)
eq_metrics_no_drag = performance_metrics(eq_returns_no_drag, eq_cum_no_drag)
spy_metrics = performance_metrics(spy_returns, spy_cum)
rsp_metrics = performance_metrics(rsp_returns, rsp_cum)
# Calculate Drawdowns (based on aligned cumulative returns)
spy_drawdown = calculate_drawdown(spy_cum_aligned)
rsp_drawdown = calculate_drawdown(rsp_cum_aligned)
eq_drag_drawdown = calculate_drawdown(eq_cum_aligned_realisitc)
eq_no_drag_drawdown = calculate_drawdown(eq_cum_aligned_no_drag)
#Alignment of drawdown series to ensure identical indices for plotting to avoid dimension errors
all_drawdown_indices = spy_drawdown.index.union(rsp_drawdown.index).union(eq_drag_drawdown.index).union(eq_no_drag_drawdown.index)
final_common_drawdown_index = all_drawdown_indices.intersection(spy_drawdown.index).intersection(rsp_drawdown.index).intersection(eq_drag_drawdown.index).intersection(eq_no_drag_drawdown.index)

spy_drawdown = spy_drawdown.loc[final_common_drawdown_index]
rsp_drawdown = rsp_drawdown.loc[final_common_drawdown_index]
eq_drag_drawdown = eq_drag_drawdown.loc[final_common_drawdown_index]
eq_no_drag_drawdown = eq_no_drag_drawdown.loc[final_common_drawdown_index]
#Data structure for drawdown plot loop
drawdown_data = {
    "SPY (Market-Cap)": (spy_drawdown, 'blue', 0.1),
    "RSP (Equal-Weight ETF)": (rsp_drawdown, 'darkorange', 0.1),
    "Artificial EW (Drag)": (eq_drag_drawdown, 'red', 0.05),
    "Artificial EW (No Drag)": (eq_no_drag_drawdown, 'green', 0.05)
}
#Create the final metrics for display
metrics_df = pd.DataFrame({
    "SPY (Market-Cap)": pd.Series(spy_metrics),
    "RSP (ETF)": pd.Series(rsp_metrics),
    "Artificial EW (Drag)": pd.Series(eq_metrics_realisitc),
    "Artificial EW (No Drag)": pd.Series(eq_metrics_no_drag)
})

# Display the performance results
print("\n--- Performance Metrics (Fixed Backtest) ---\n")
print(metrics_df.to_markdown(floatfmt=".4f"))

#Plotting Cumulative Returns
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14,7))
plt.plot(spy_cum_aligned, label="SPY (Market-Cap)")
plt.plot(rsp_cum_aligned, label="RSP (Equal-Weight ETF)")
plt.plot(eq_cum_aligned_realisitc, label="Artificial Equal-Weight (Drag)")
plt.plot(eq_cum_aligned_no_drag, label="Artificial Equal-Weight (No Drag)")
plt.title("Backtest: SPY vs RSP vs Artificial Equal-Weight (Drag) vs Artificial Equal-Weight (No Drag)")
plt.xlabel("Date")
plt.ylabel("Cumulative Growth ($1 invested)")
plt.legend()
plt.grid(True)
plt.show()

#Plotting Max Drawdown
plt.style.use('seaborn-v0_8-whitegrid') 
plt.figure(figsize=(14, 7))

#Plot the four drawdown series AND fill the area using a simple loop structure
for label, (drawdown_series, color, alpha) in drawdown_data.items():
    x_data = drawdown_series.index.values.ravel() #Array of Dates
    y_data = drawdown_series.values.ravel()       #Array of Drawdown Values
    
    #Create a clean 1D boolean mask array
    mask = (~np.isnan(y_data)).astype(bool)
    
    #Apply the mask to both arrays to get clean data
    x_data_clean = x_data[mask]
    y_data_clean = y_data[mask]
    
    #Plot the line and fill
    plt.plot(x_data_clean, y_data_clean, label=label, color=color)
    plt.fill_between(x_data_clean, 
                     y_data_clean, 
                     0, 
                     color=color, 
                     alpha=alpha)


plt.title("Backtest: Portfolio Drawdown Comparison (2018–2025)")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")

# Format the y-axis to show percentages
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
plt.legend()
plt.grid(True)
plt.show()