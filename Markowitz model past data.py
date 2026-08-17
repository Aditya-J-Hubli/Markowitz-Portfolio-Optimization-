import numpy as np 
import yfinance as yf 
import matplotlib.pyplot as plt

tickers = [
    'ADANIGREEN.NS', 
    'TCS.NS',
    'HDFCBANK.NS',
    'INFY.NS',
    'GOLDBEES.NS'
]
asset_names = ['Adani Green Energy', 'TCS', 'HDFC Bank', 'Infosys', 'Gold ETF']
print('Downloading data : ')
raw = yf.download(tickers,
                  start= '2021-01-01', 
                  end= '2024-12-31',
                  auto_adjust= True, 
                  progress= False)['Close']
print(f"\nData shape : {raw.shape} ({raw.shape[0]} trading days, {raw.shape[1]} assets)")
print('Missing values per ticker : \n', raw.isna().sum())

# drop rows where any tickers has mising data 
raw = raw.dropna()
print(f"After dropping NaN rows : {raw.shape[0]} trading days remain")

# log returns 
log_returns = np.log(raw / raw.shift(1)).dropna()
print(f"Return shape : {log_returns.shape}")

# annual mu and sigma 
t_days = 252
mu = log_returns.mean().values * t_days
sigma = log_returns.cov().values * t_days

print('\nAnnualised expected returns : ')
for name, m in zip(asset_names, mu):
    print(f"{name:12s}: {m:.2%}")

print('\nAnnualised covarinace matrix (diagonal == variances) : ')
print(np.round(sigma, 6))

# verify Eigenvalues 
print('Sigma Eigenvalues : ', np.round(np.linalg.eigvalsh(sigma), 6))
print('All +ve matrix ? ', np.all(np.linalg.eigvalsh(sigma) > 0))

# Markowitz function 
def markowitz_weights(mu, sigma, mu_target):
    n = len(mu)
    ones = np.ones(n)
    sigma_inv = np.linalg.inv(sigma)

    A = float(ones.T @ sigma_inv @ ones) # total inverse risk in the universe 
    B = float(ones.T @ sigma_inv @ mu) # Risk-adjusted return portfolio 
    C = float(mu.T @ sigma_inv @ mu) # pure return-per-risk measure 
    D = A * C - B**2 # 'Room' between two constraints 

    #Lagrange multipliers 
    lambda1 = (A * mu_target - B) / D
    lambda2 = (C - B * mu_target) / D

    # optimal weights 
    w = sigma_inv @ (lambda1 * mu + lambda2 * ones)
    return w 

# single optimal portfolio 
mu_min = mu.min()
mu_max = mu.max()
mu_target = (mu_min + mu_max) / 2 # calculate the average, midpoint for feseable range 
print(f"Target Return chosen : {mu_target:.2%}")

weight_optimization = markowitz_weights(mu, sigma, mu_target)
port_return         = weight_optimization @ mu
port_var            = weight_optimization @ sigma @ weight_optimization
port_vol            = np.sqrt(port_var)

print('\n Optimal Weights: ')
for name, w in zip(asset_names, weight_optimization):
    print(f" {name:12s} : {w:+.4f} ({w:+.1%})")

print(f"\nAchieved return : {port_return:.4%}")
print(f"Portfolio std dev : {port_vol:.4%}")
print(f"weights sum to 1 : {np.sum(weight_optimization):.6f}")

# efficient frontier 
n_points = 200
target_returns = np.linspace(mu_min * 1.01, mu_max * 0.99, n_points)
frontier_rets = []
frontier_vols = []

for mu_t in target_returns:
    try:
        w       = markowitz_weights(mu, sigma, mu_t) #solve the optimal weights, find the safest portfolio for this return 
        sigma_port = np.sqrt(w @ sigma @ w) #measure the risk (volatility) for that portfolio  
        frontier_vols.append(sigma_port) # store the risk, remember the data points 
        frontier_rets.append(mu_t) # store the return, pair it with its returns 
    except Exception:
        continue
# After 200 iterations: 200 (risk, return) pairs → plot them → efficient frontier

# PLOT 
fig, ax = plt.subplots(figsize = (10, 7))

ax.plot(frontier_vols, frontier_rets, color = '#378ADD', linewidth = 2.5, 
        label = 'Efficient Frontier') 

# individual assets 
colors = ['#D85A30', '#1D9E75', '#7F77DD', '#EF9F27', '#888780'] 
for i, name in enumerate(asset_names):
    asset_vol = np.sqrt(sigma[i, i])
    ax.scatter(asset_vol, mu[i], s=100, color = colors[i], zorder = 5)
    ax.annotate(name, (asset_vol, mu[i]), textcoords = 'offset points', xytext = (8, 4),
                 fontsize = 10)

# optimal portfolio point 
ax.scatter(port_vol, port_return, color = 'red', s=150, zorder = 6, 
           marker = '*', label = f"optimal portfolio ({mu_target:.1%} target)")

ax.set_xlabel('Portfolio risk/volatility (annualised)', fontsize = 10)
ax.set_ylabel('Expected Portfolio return (annualised)', fontsize = 10)
ax.set_title('MARKOWITZ EFFICIENT FRONTIER ~ INDIAN MARKET', fontsize = 15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
ax.legend(fontsize = 10)
ax.grid(alpha = 0.3)
plt.tight_layout()
plt.savefig('efficient_frontier_IND.png', dpi = 150)
plt.show()
print('\nPlot saved as efficient_frontier_IND.png')



