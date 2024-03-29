#
# Monte Carlo valuation of European call options with NumPy (log version)
# mcs_full_vector_numpy.py
#
import math
from numpy import *
from time import time

# star import for shorter code

random.seed(20000)
t0 = time()

# Parameters
S0 = 100.;
K = 105.;
T = 1.0;
r = 0.05;
sigma = 0.2
M = 100;
dt = T / M;
I = 250000 * 1

# Simulating I paths with M time steps
ran = random.standard_normal((M + 1, I))
ran -= ran.mean()  # corrects 1st moment
ran /= ran.std()  # corrects 2nd moment
S = zeros_like(ran)
S[0] = S0
S[1:] = S0 * exp(cumsum((r - 0.5 * sigma ** 2) * dt
                        + sigma * math.sqrt(dt) * ran[1:], axis=0))
# sum instead of cumsum would also do
# if only the final values are of interest


# Calculating the Monte Carlo estimator
C0 = math.exp(-r * T) * sum(maximum(S[-1] - K, 0)) / I

# Results output
tnp2 = time() - t0
print("European Option Value %7.3f" % C0)
print("Duration in Seconds   %7.3f" % tnp2)

import matplotlib.pyplot as plt

plt.plot(S[:, :1000])
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('index level')
# tag: index_paths
# title: The first 10 simulated index level paths
plt.show()
