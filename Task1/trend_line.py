import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import trendln

import yfinance as yf  # requires yfinance - pip install yfinance

tick = yf.Ticker('^GSPC')  # S&P500
hist = tick.history(period="max", rounding=True)
h = hist[-1000:].Close

#
# mins, maxs = trendln.calc_support_resistance(h)
# minimaIdxs, pmin, mintrend, minwindows = trendln.calc_support_resistance((hist[-1000:].Low, None)) #support only
# mins, maxs = trendln.calc_support_resistance((hist[-1000:].Low, hist[-1000:].High))
# (minimaIdxs, pmin, mintrend, minwindows), (maximaIdxs, pmax, maxtrend, maxwindows) = mins, maxs


data = hist[-100:]
idx = data.index
fig = trendln.plot_sup_res_date((data.Low, data.High), idx)  # requires pandas
plt.savefig('suppres.svg', format='svg')
plt.show()
plt.clf()  # clear figure
