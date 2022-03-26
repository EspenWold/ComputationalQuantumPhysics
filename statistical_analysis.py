import os
import numpy as np
from time import time
import pandas as pd

DATA_ID = "Results/EnergyMin"


def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)


infile = open(data_path("Energies.dat"), 'r')


def bootstrap(data, bootstrap_data_points):
    t = np.zeros(bootstrap_data_points)
    n = len(data)
    t0 = time()

    for i in range(bootstrap_data_points):
        # Resample data
        t[i] = np.mean(data[np.random.randint(0, n, n)])

    # analysis
    print("Runtime: %g sec" % (time() - t0))
    print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %8g %14g %15g" % (np.mean(data), np.std(data), np.mean(t), np.std(t)))
    return t


def time_series_bootstrap(data, statistic, r, l):
    t = np.zeros(r)
    n = len(data)
    k = int(np.ceil(float(n) / l))
    t0 = time()

    # time series bootstrap
    for i in range(r):
        # construct bootstrap sample from
        # k chunks of data. The chunksize is l
        _data = np.concatenate([data[j:j + l] for j in np.random.randint(0, n - l, k)])[0:n]
        t[i] = statistic(_data)

    # analysis
    print("Runtime: %g sec" % (time() - t0))
    print("Bootstrap Statistics :")
    print("original           bias      std. error")
    print("%8g %14g %15g" % (statistic(data),  np.mean(t) - statistic(data), np.std(t)))
    return t


def block(data):
    # preliminaries
    n = len(data)
    d = int(np.log2(n))
    s, gamma = np.zeros(d), np.zeros(d)
    mu = mean(data)

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in np.arange(0, d):
        n = len(data)
        # estimate autocovariance of x
        gamma[i] = n ** (-1) * sum((data[0:(n - 1)] - mu) * (data[1:n] - mu))
        # estimate variance of x
        s[i] = var(data)
        # perform blocking transformation
        data = 0.5 * (data[0::2] + data[1::2])

    # generate the test observator M_k from the theorem
    m = (np.cumsum(((gamma / s) ** 2 * 2 ** np.arange(1, d + 1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = np.array(
        [6.634897, 9.210340, 11.344867, 13.276704, 15.086272, 16.811894, 18.475307, 20.090235, 21.665994, 23.209251,
         24.724970, 26.216967, 27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306, 36.190869, 37.566235,
         38.932173, 40.289360, 41.638398, 42.979820, 44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in np.arange(0, d):
        if m[k] < q[k]:
            break
    if k >= d - 1:
        print("Warning: Use more data")
    return mu, s[k] / 2 ** (d - k)


x = np.loadtxt(infile)
(mean, var) = block(x)
std = np.sqrt(var)

results = {'Mean': [mean], 'STDev': [std]}
frame = pd.DataFrame(results, index=['Values'])
print(frame)
