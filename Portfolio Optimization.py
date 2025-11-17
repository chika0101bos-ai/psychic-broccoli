"""
Preface:

What this script does:
- Downloads historical adjusted close prices for a list of tickers using yfinance.
- Computes daily returns, annualized expected returns, and covariance matrices.
- Runs three portfolio optimization methods using PyPortfolioOpt:
    1. Max Sharpe (Efficient Frontier)
    2. Hierarchical Risk Parity (HRP)
    3. Conditional Value at Risk (CVaR)
- Performs discrete allocation based on a total portfolio value.
- Saves all outputs (prices, returns, weights, allocations, plots, and summary) to a designated folder.

Requirements to run:
- Python 3.8+
- Libraries: numpy, pandas, yfinance, matplotlib, PyPortfolioOpt
- Internet connection (to download price data)
- Optional: the tickers you want to analyze and the date range.

Notes:
- The script assumes all required libraries are installed and that the tickers exist.
- All outputs are written to a folder called 'portfolio_outputs'.
- For a “production-safe” version with warnings and fallbacks, see the ARCHIVED SAFETY LOGIC at the end of the script.
"""
import sys
import os
from typing import List, Dict
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier, EfficientCVaR
from pypfopt.discrete_allocation import DiscreteAllocation
from pypfopt import HRPOpt


TICKERS = ["MRNA", "PFE", "JNJ", "GOOGL", "META", "AAPL", "COST", "WMT", "KR", "JPM", "BAC", "HSBC"]
START = dt.datetime(2019, 9, 15)
END = dt.datetime(2021, 9, 15)
TOTAL_PORTFOLIO_VALUE = 100_000
RISK_FREE_RATE = 0.02
OUTDIR = "portfolio_outputs"
os.makedirs(OUTDIR, exist_ok=True)



def download_prices(tickers: List[str], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    prices = pd.DataFrame()
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
        prices[t] = df["Close"]
    prices.index.name = "Date"
    return prices.sort_index()

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def safe_mean_and_cov(prices: pd.DataFrame):
    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    return mu, S

def to_csv_safe(df: pd.DataFrame, filename: str):
    path = os.path.join(OUTDIR, filename)
    df.to_csv(path, index=True)
    print(f"Saved: {path}")



def run_max_sharpe(mu: pd.Series, S: pd.DataFrame, price_df: pd.DataFrame):
    ef = EfficientFrontier(mu, S)
    ef.max_sharpe()
    cleaned = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=RISK_FREE_RATE)

    latest_prices = price_df.iloc[-1]
    da = DiscreteAllocation(cleaned, latest_prices, total_portfolio_value=TOTAL_PORTFOLIO_VALUE)
    allocation, leftover = da.greedy_portfolio()

    return {"weights": cleaned, "performance": perf, "allocation": allocation, "leftover": leftover}

def run_hrp(price_df: pd.DataFrame):
    returns = price_df.pct_change().dropna()
    hrp = HRPOpt(returns)
    hrp_weights = hrp.optimize()
    perf = hrp.portfolio_performance(verbose=False)

    latest_prices = price_df.iloc[-1]
    da = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value=TOTAL_PORTFOLIO_VALUE)
    allocation, leftover = da.greedy_portfolio()

    return {"weights": hrp_weights, "performance": perf, "allocation": allocation, "leftover": leftover}

def run_cvar(mu: pd.Series, returns: pd.DataFrame, price_df: pd.DataFrame):
    ef_cvar = EfficientCVaR(mu, returns)
    ef_cvar.min_cvar()
    cleaned = ef_cvar.clean_weights()
    perf = ef_cvar.portfolio_performance(verbose=False)

    latest_prices = price_df.iloc[-1]
    da = DiscreteAllocation(cleaned, latest_prices, total_portfolio_value=TOTAL_PORTFOLIO_VALUE)
    allocation, leftover = da.greedy_portfolio()

    return {"weights": cleaned, "performance": perf, "allocation": allocation, "leftover": leftover}



def plot_weights(weights_dict: Dict[str, float], title: str, fname: str = None):
    items = sorted(weights_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, vals)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    if fname:
        path = os.path.join(OUTDIR, fname)
        plt.savefig(path)
        print(f"Saved plot: {path}")
    else:
        plt.show()
    plt.close()



def main(tickers=TICKERS, start=START, end=END):
    print("Downloading prices for:", tickers)
    price_df = download_prices(tickers, start, end).ffill().bfill()
    to_csv_safe(price_df, "prices.csv")

    returns = compute_returns(price_df)
    to_csv_safe(returns, "returns.csv")

    mu, S = safe_mean_and_cov(price_df)
    mu.to_csv(os.path.join(OUTDIR, "mu_annualized.csv"))
    S.to_csv(os.path.join(OUTDIR, "cov_annualized.csv"))

    results = {}

    print("\n=== Running Max Sharpe ===")
    max_sharpe_res = run_max_sharpe(mu, S, price_df)
    results["max_sharpe"] = max_sharpe_res
    pd.Series(max_sharpe_res["weights"]).to_csv(os.path.join(OUTDIR, "weights_max_sharpe.csv"))
    pd.DataFrame.from_dict(max_sharpe_res["allocation"], orient="index", columns=["shares"]).to_csv(
        os.path.join(OUTDIR, "allocation_max_sharpe.csv"))
    plot_weights(max_sharpe_res["weights"], "Max Sharpe Allocation", "weights_max_sharpe.png")

    print("\n=== Running HRP ===")
    hrp_res = run_hrp(price_df)
    results["hrp"] = hrp_res
    pd.Series(hrp_res["weights"]).to_csv(os.path.join(OUTDIR, "weights_hrp.csv"))
    pd.DataFrame.from_dict(hrp_res["allocation"], orient="index", columns=["shares"]).to_csv(
        os.path.join(OUTDIR, "allocation_hrp.csv"))
    plot_weights(hrp_res["weights"], "HRP Allocation", "weights_hrp.png")

    print("\n=== Running CVaR ===")
    cvar_res = run_cvar(mu, returns, price_df)
    results["cvar"] = cvar_res
    pd.Series(cvar_res["weights"]).to_csv(os.path.join(OUTDIR, "weights_cvar.csv"))
    pd.DataFrame.from_dict(cvar_res["allocation"], orient="index", columns=["shares"]).to_csv(
        os.path.join(OUTDIR, "allocation_cvar.csv"))
    plot_weights(cvar_res["weights"], "CVaR Allocation", "weights_cvar.png")

    summary = []
    for method, r in results.items():
        w = r["weights"]
        perf = r["performance"]
        leftover = r["leftover"]
        num_positions = len([v for v in w.values() if v > 0])
        summary.append({"method": method, "performance": str(perf),"leftover": leftover, "num_positions": num_positions})
    summary_df = pd.DataFrame(summary)
    to_csv_safe(summary_df, "optimization_summary.csv")
    print("\nDone. Outputs written to:", os.path.abspath(OUTDIR))


if __name__ == "__main__":
    main()


# ARCHIVED SAFETY LOGIC
"""
# Original warning/fallback imports and HAS_PFOPT/HAS_EF_CVAR flags
import warnings

HAS_PFOPT = False
HAS_EF_CVAR = False
try:
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt import HRPOpt
    try:
        from pypfopt.efficient_frontier import EfficientCVaR
        HAS_EF_CVAR = True
    except Exception:
        HAS_EF_CVAR = False
    HAS_PFOPT = True
except Exception:
    warnings.warn(
        "PyPortfolioOpt not fully available. Some optimization methods (Efficient Frontier, "
        "HRP, CVaR, DiscreteAllocation) will be skipped. Install with: pip install PyPortfolioOpt"
    )
    HAS_PFOPT = False
    HAS_EF_CVAR = False

# NaN and fallback handling in price downloads and returns
if df.empty:
    warnings.warn(f"No data for {t}; skipping.")
rets = prices.pct_change().dropna()
mu = rets.mean() * 252
S = rets.cov() * 252

# Defensive coding in optimizations (try/excepts)
try:
    ef.max_sharpe()
except Exception as e:
    print("Max Sharpe failed:", e)
    return None

try:
    da = DiscreteAllocation(cleaned, latest_prices, total_portfolio_value=TOTAL_PORTFOLIO_VALUE)
    allocation, leftover = da.greedy_portfolio()
except Exception as e:
    warnings.warn(f"Discrete allocation failed: {e}")
    allocation, leftover = {}, 0.0

# CVaR compatibility try/except
try:
    ef_cvar = EfficientCVaR(mu, returns)
    ef_cvar.min_cvar()
except Exception:
    try:
        ef_cvar = EfficientCVaR(mu, returns)
        ef_cvar.min_cvar()
    except Exception as e:
        warnings.warn("EfficientCVaR failed: " + str(e))
        return None

# Defensive latest_prices NaN checks
if latest_prices.isna().any():
    raise ValueError("NaNs present in latest_prices even after cleaning.")
"""
