# Performance & Cost-Drag Analysis of S&P 500 Factor Portfolios

A comprehensive empirical study comparing Equal-Weight vs Market-Cap Weighted S&P 500 portfolios, with detailed analysis of transaction costs and real-world implementation inefficiencies.

## Research Question

**To what extent does implementation "drag" negate the performance premium of a quarterly rebalanced Equal-Weight S&P 500 portfolio compared to the standard Market-Capitalization Weighted Index (SPY)?**

**Period**: January 2018 - October 2025 (7.75 years)

## Key Findings

### Performance Summary

| Portfolio | Total Return | Sharpe Ratio | Max Drawdown | Annualized Return |
|-----------|--------------|--------------|--------------|-------------------|
| **Artificial EW (No Drag)** | **199.86%** | **0.7601** | -38.39% | 15.21% |
| **Artificial EW (Drag)** | 191.83% | 0.7400 | -38.41% | 14.80% |
| **SPY (Market-Cap)** | 178.69% | 0.7193 | **-33.72%** | 14.12% |
| **RSP (Actual ETF)** | 110.63% | 0.5022 | -39.04% | 10.08% |

### Critical Insights

1. **Cost Impact Quantified**: The 0.35% annual transaction cost reduced total gains by **8.03%** over the 7.75-year period due to compounding effects.

2. **Implementation Gap**: The actual RSP ETF underperformed its theoretical model by **~81%** in total returns, revealing significant real-world execution inefficiencies beyond simple cost assumptions.

3. **Risk Trade-off**: Equal-weight strategies showed **15-17% higher drawdowns** than market-cap weighted approaches, reflecting greater exposure to smaller-cap volatility.

## Methodology

### Portfolio Constructions

1. **SPY (Benchmark)** - Standard S&P 500 Market-Cap Weighted ETF
2. **RSP (Reality Check)** - Actual S&P 500 Equal-Weight ETF
3. **Artificial EW with Drag** - Simulated portfolio with 0.55% annual cost
   - Expense Ratio: 0.20%
   - Transaction Cost: 0.35%
4. **Artificial EW (No Drag)** - Theoretical best-case with 0.20% expense ratio only

### Data & Implementation

- **Data Source**: Yahoo Finance (Adjusted Close Prices)
- **Universe**: Current S&P 500 constituents (survivorship bias noted)
- **Rebalancing**: Quarterly (every 63 trading days)
- **Assumptions**: 
  - 0% risk-free rate for Sharpe calculations
  - 0.35% transaction cost based on realistic turnover estimates
  - Equal weighting across all available tickers at rebalance

### Robustness

The dual simulation approach (with/without drag) against the actual RSP ETF provides validation:
- Simulated models confirm theoretical equal-weight premium
- RSP's underperformance reveals practical implementation challenges
- Cost sensitivity clearly demonstrated through drag comparison

## Performance Metrics Explained

### Total Return
```
Total Return = (Portfolio_End / Portfolio_Start) - 1
```
Cumulative gain/loss over entire period.

### Annualized Return
```
Annualized Return = (1 + Total Return)^(252/Days) - 1
```
Constant annual rate equivalent to total return.

### Volatility
```
Volatility = StDev(Daily Returns) × √252
```
Annualized standard deviation of daily returns.

### Sharpe Ratio
```
Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Volatility
```
Risk-adjusted return (assumes 0% risk-free rate).

### Maximum Drawdown
```
Max Drawdown = Min((Portfolio Value / Highest Portfolio Value) - 1)
```
Largest peak-to-trough decline.

## Visualizations

The analysis generates two key plots:

### 1. Cumulative Returns (Log Scale)
- Tracks $1 invested across all four portfolios
- Highlights COVID-19 crash (2020) and recovery
- Shows 2022 drawdown from rising rates/inflation
- Demonstrates 2023-2024 large-cap tech outperformance in SPY

### 2. Drawdown Comparison
- Visualizes peak-to-trough declines over time
- Filled areas show magnitude of losses
- Market-cap weighting shows superior downside protection
- Equal-weight strategies exhibit higher volatility risk

## Key Observations

### Period-Specific Behavior

**2020 (COVID Crash)**
- All portfolios experienced significant drawdowns
- Recovery patterns diverged based on weighting scheme

**2022 (Rate Hikes)**
- Rising interest rates and inflation impacted all strategies
- RSP/SPY overlap indicates similar stress response

**2023-2024 (Tech Rally)**
- SPY outperformed due to concentrated large-cap tech exposure
- Equal-weight strategies missed some Magnificent 7 gains
- Demonstrates concentration risk trade-offs

### Cost Sensitivity

The 8.03% performance degradation from just 0.35% annual costs highlights:
- Compounding impact over multi-year periods
- Critical importance of low-cost execution platforms
- Need to account for slippage, spreads, and market impact

### Implementation Reality

RSP's dramatic underperformance suggests unmeasured costs:
- Market impact from large rebalancing trades
- Timing differences (daily vs. quarterly rebalancing)
- Operational inefficiencies
- Potential tracking error from fund flows

## Limitations & Caveats

### Data Considerations
- **Survivorship Bias**: Using current S&P 500 constituents may overstate performance
- **Historical Data**: Includes historical prices for tickers not in index during entire period
- **Corporate Actions**: Adjusted close prices account for splits/dividends

### Model Assumptions
- **Transaction Costs**: 0.35% is an estimate; actual costs vary by broker and market conditions
- **Market Impact**: Large institutional trades face additional slippage not modeled
- **Tax Efficiency**: Tax implications of frequent rebalancing not considered
- **Rebalancing Timing**: Assumes perfect execution at close prices

### Real-World Factors Not Modeled
- Bid-ask spreads
- Partial fills and order routing
- Market impact from large orders
- Cash drag from dividends
- Fund flows and redemptions (for RSP)

## Statistical Notes

### Risk-Free Rate Assumption
The analysis assumes 0% risk-free rate for Sharpe ratio calculations. During 2018-2025:
- Fed rates ranged from near-zero (2020-2021) to 5%+ (2023-2024)
- Using actual rates would lower Sharpe ratios proportionally
- Relative comparisons remain valid

### Volatility Differences
Equal-weight strategies showed only 0.36% higher volatility than SPY despite:
- Greater exposure to smaller-cap stocks
- Less concentration in mega-cap tech
- This suggests diversification benefits partially offset size-factor volatility

## References

### Data Sources
- **Yahoo Finance**: Historical price data and ETF information
- **Wikipedia**: S&P 500 constituent list
- **S&P Dow Jones Indices**: Index methodology
- **Invesco**: RSP ETF specifications

### Learning Resources
- freeCodeCamp.org: "Algorithmic Trading Using Python - Full Course"
  - [YouTube Tutorial](https://www.youtube.com/watch?v=xfzGZB4HhEE)

## Author

- GitHub: https://github.com/Vonoa
- LinkedIn: https://www.linkedin.com/in/aled-von-oppell-40b315289/

## Acknowledgments

- S&P Dow Jones Indices for methodology documentation
- Yahoo Finance for accessible market data
- Invesco for RSP ETF transparency
- The quantitative finance community for best practices
