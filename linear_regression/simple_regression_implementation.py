from statistics import mean
import numpy as np

def best_fit_slope(xs, ys):
	m = ( (mean(xs) * mean(ys) - mean(xs * ys)) /
		  ((mean(xs)**2) - mean(xs**2)) )
	return m

xs = np.array([1,2,3,4,5])
ys = np.array([5,4,6,5,6])

m = best_fit_slope(xs, ys)
print(m)