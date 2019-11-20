from numpy import *

S0 = 100  # underlying
K = 105.  # Stirke
T = 1.0  # Time to expiry
r = 0.05  # riskless short rate(interest)
sigma = 0.2  # vola in %

I = 100000  # iterations
z = random.standard_normal(I)
ST = S0 * exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * z)
hT = maximum(ST - K, 0)
C0 = exp(-r * T) * sum(hT) / I

print("Value of European call : %5.3F" % C0)
