from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

def best_fit_line(xs, ys):
	m = ( (mean(xs) * mean(ys) - mean(xs * ys)) /
		  ((mean(xs)**2) - mean(xs**2)) )
	b = mean(ys) - m * mean(xs)
	return m, b

xs = np.array([1,2,3,4,5])
ys = np.array([5,4,6,5,6])

m, b = best_fit_line(xs, ys)

regression_line = [m*x+b for x in xs]
plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()