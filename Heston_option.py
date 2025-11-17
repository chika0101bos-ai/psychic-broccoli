"""
Preface

What this script does:
- Simulates tick-level SPY prices over a defined period.
- Uses Monte Carlo to price European call options under the Heston stochastic volatility model.
- Computes key option Greeks (delta, gamma, vega, theta, rho) at each tick.
- Performs a delta-hedging simulation across a sequence of ticks, tracking:
    - Option value
    - Target hedge shares
    - Shares traded
    - Transaction costs
    - Daily P&L
- Outputs a hedging log as a pandas DataFrame.

Requirements to run:
- Python 3.8+
- Libraries: numpy, pandas, numba
- Optional: interactive environment (Jupyter, VS Code) for exploring full logs

Notes:
- All Monte Carlo simulations use fixed random seeds for reproducibility.
- The script currently simulates a single-day, short-maturity option.
- All outputs are in-memory (`hedge_df`) and can be saved to CSV.
- For a more “production-safe” version with warnings and fallbacks, see ARCHIVED SAFETY LOGIC at the end.
"""

import numpy as np
import pandas as pd
from numba import jit

np.random.seed(42)
DATES = pd.date_range("2020-11-16", periods=1000, freq="5min")
mid_price = 338 + np.cumsum(np.random.randn(len(DATES)) * 0.2)
bid_price = mid_price - 0.017
ask_price = mid_price + 0.017
tick_data = pd.DataFrame({'timestamp': DATES, 'mid_price': mid_price, 'bid_price': bid_price, 'ask_price': ask_price})
SPY_S0 = tick_data['mid_price'].iloc[0]



@jit(nopython=True, fastmath=True)
def heston_mc(S0, v0, r, kappa, theta, sigma, rho, T, M, N, Z1, Z2):
    dt = T / M
    S = np.zeros((M+1, N))
    v = np.zeros((M+1, N))
    S[0, :] = S0
    v[0, :] = v0
    for t in range(1, M+1):
        v_prev = v[t-1, :]
        S_prev = S[t-1, :]
        v_new = v_prev + kappa*(theta - np.maximum(v_prev,0))*dt + sigma*np.sqrt(np.maximum(v_prev,0))*np.sqrt(dt)*Z2[t-1,:]
        v_new = np.maximum(v_new, 0.0)
        S_new = S_prev * np.exp((r - 0.5*v_prev)*dt + np.sqrt(v_prev*dt)*Z1[t-1,:])
        S[t, :] = S_new
        v[t, :] = v_new
    return S, v

def heston_call_price_mc(S0, K, T, r, v0, kappa, theta, sigma, rho, N_paths=10000, M_steps=50):
    np.random.seed(42)
    Z1 = np.random.standard_normal((M_steps, N_paths))
    Z2 = rho*Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal((M_steps, N_paths))
    S_paths, _ = heston_mc(S0, v0, r, kappa, theta, sigma, rho, T, M_steps, N_paths, Z1, Z2)
    payoff = np.maximum(S_paths[-1, :] - K, 0)
    return np.exp(-r*T) * np.mean(payoff)



def heston_greeks_mc(S0, K, T, r, v0, kappa, theta, sigma, rho, N_paths=10000, M_steps=50,eps_S=0.01, eps_v=0.001, eps_T=1/252/24/12, eps_r=1e-4):
    # price at spot
    C0 = heston_call_price_mc(S0, K, T, r, v0, kappa, theta, sigma, rho, N_paths, M_steps)
    # delta, gamma, vega (sensitivity to variance), theta (sensitivity to time decay),rho (sensitivity to interest rate) ->respenctively
    C_up = heston_call_price_mc(S0 + eps_S, K, T, r, v0, kappa, theta, sigma, rho, N_paths, M_steps)
    C_down = heston_call_price_mc(S0 - eps_S, K, T, r, v0, kappa, theta, sigma, rho, N_paths, M_steps)
    delta = (C_up - C_down) / (2 * eps_S)
    gamma = (C_up - 2*C0 + C_down) / (eps_S**2)
    C_vol_up = heston_call_price_mc(S0, K, T, r, v0 + eps_v, kappa, theta, sigma, rho, N_paths, M_steps)
    C_vol_down = heston_call_price_mc(S0, K, T, r, v0 - eps_v, kappa, theta, sigma, rho, N_paths, M_steps)
    vega = (C_vol_up - C_vol_down) / (2 * eps_v)
    if T > eps_T:
        C_T_minus = heston_call_price_mc(S0, K, T - eps_T, r, v0, kappa, theta, sigma, rho, N_paths, M_steps)
        theta = (C_T_minus - C0) / eps_T
    else:
        theta = 0.0
    C_r_up = heston_call_price_mc(S0, K, T, r + eps_r, v0, kappa, theta, sigma, rho, N_paths, M_steps)
    C_r_down = heston_call_price_mc(S0, K, T, r - eps_r, v0, kappa, theta, sigma, rho, N_paths, M_steps)
    rho_val = (C_r_up - C_r_down) / (2 * eps_r)
    return delta, gamma, vega, theta, rho_val



OPTION_STRIKE = SPY_S0
OPTION_MATURITY_YEARS = 1/252 
RISK_FREE_RATE = 0.01
V0_EST = 0.04
H_KAPPA = 2.0
H_THETA = 0.04
H_SIGMA = 0.3
H_RHO = -0.7
H_N_PATHS = 2000
H_M_STEPS = 50



heston_price = heston_call_price_mc(SPY_S0, OPTION_STRIKE, OPTION_MATURITY_YEARS, RISK_FREE_RATE, V0_EST, H_KAPPA, H_THETA, H_SIGMA, H_RHO, H_N_PATHS, H_M_STEPS)
delta, gamma, vega, theta, rho_val = heston_greeks_mc(SPY_S0, OPTION_STRIKE, OPTION_MATURITY_YEARS, RISK_FREE_RATE, V0_EST, H_KAPPA, H_THETA, H_SIGMA, H_RHO, H_N_PATHS, H_M_STEPS)



option_quantity = 100  
portfolio_cash = 0.0
current_option_value = heston_price * option_quantity
current_hedge_shares = -option_quantity * delta
portfolio_cash -= current_hedge_shares * SPY_S0

SIMULATION_TICKS = 500
simulation_df = tick_data.tail(SIMULATION_TICKS).copy()
prev_spot_price = SPY_S0
prev_hedge_shares = current_hedge_shares
prev_option_value = current_option_value

hedge_log = []

for i, row in simulation_df.iterrows():
    current_spot_price = row['mid_price']
    new_option_value = heston_call_price_mc(current_spot_price, OPTION_STRIKE, OPTION_MATURITY_YEARS, RISK_FREE_RATE,V0_EST, H_KAPPA, H_THETA, H_SIGMA, H_RHO, H_N_PATHS, H_M_STEPS) * option_quantity
    delta, gamma, vega, theta_val, rho_val = heston_greeks_mc(current_spot_price, OPTION_STRIKE, OPTION_MATURITY_YEARS,RISK_FREE_RATE, V0_EST, H_KAPPA, H_THETA, H_SIGMA, H_RHO,H_N_PATHS, H_M_STEPS)
    target_hedge_shares = -option_quantity * delta
    shares_to_trade = target_hedge_shares - prev_hedge_shares
    # Transaction costs (bid/ask)
    transaction_cost = 0
    if shares_to_trade > 0:
        portfolio_cash -= shares_to_trade * row['ask_price']
        transaction_cost = shares_to_trade * (row['ask_price'] - current_spot_price)
    elif shares_to_trade < 0:
        portfolio_cash -= shares_to_trade * row['bid_price']
        transaction_cost = -shares_to_trade * (current_spot_price - row['bid_price'])
    
    pnl_from_option_change = (prev_option_value - new_option_value)
    pnl_from_underlying = prev_hedge_shares * (current_spot_price - prev_spot_price)
    daily_pnl = pnl_from_option_change + pnl_from_underlying - transaction_cost
    
    prev_spot_price = current_spot_price
    prev_hedge_shares = target_hedge_shares
    prev_option_value = new_option_value
    
    hedge_log.append({'timestamp': row['timestamp'],'spot_price': current_spot_price,'option_value': new_option_value,'delta': delta,'gamma': gamma,'vega': vega,'theta': theta_val,'rho': rho_val,'target_hedge_shares': target_hedge_shares,'shares_traded': shares_to_trade,'transaction_cost': transaction_cost,'daily_pnl': daily_pnl})

hedge_df = pd.DataFrame(hedge_log)
total_hedged_pnl = hedge_df['daily_pnl'].sum()

print(f"Initial Option Value: {current_option_value:.2f}")
print(f"Total Simulated Hedged P&L over {SIMULATION_TICKS} ticks: ${total_hedged_pnl:.2f}")
print("\nHedging Log (first 10 rows):")
print(hedge_df.head(10))
hedge_df.to_csv("full_hedge_log.csv", index=False)
print("Full hedging log saved to: full_hedge_log.csv")

# ARCHIVED SAFETY LOGIC
"""
# Defensive coding / safety measures originally included:

# 1. Ensure variance v >= 0 during Heston MC simulation:
v_new = np.maximum(v_prev + kappa*(theta - np.maximum(v_prev,0))*dt + sigma*np.sqrt(np.maximum(v_prev,0))*np.sqrt(dt)*Z2[t-1,:], 0.0)

# 2. Seed resets for reproducibility in MC simulations:
np.random.seed(42)

# 3. NaN and negative price handling:
# (if any NaNs in tick data, drop or forward-fill)
tick_data = tick_data.ffill().bfill().dropna()

# 4. Defensive checks in finite-difference Greek computations:
# Ensure epsilon perturbation does not produce negative prices
S0_up = max(S0 + eps, 0)
S0_down = max(S0 - eps, 0)

# 5. Hedging simulation:
# Protect against division by zero or negative shares:
shares_to_trade = np.sign(shares_to_trade) * abs(shares_to_trade)

# 6. Catch Monte Carlo errors:
try:
    S_paths, _ = heston_mc(...)
except Exception as e:
    print("Heston MC failed:", e)
    return None

# 7. Transaction cost computation protected against negative spreads
transaction_cost = max(0, computed_transaction_cost)
"""