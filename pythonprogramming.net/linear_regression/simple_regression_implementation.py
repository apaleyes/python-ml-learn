from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

def best_fit_line(xs, ys):
	m = ( (mean(xs) * mean(ys) - mean(xs * ys)) /
		  ((mean(xs)**2) - mean(xs**2)) )
	b = mean(ys) - m * mean(xs)
	return m, b

def squared_error(ys_orig, ys_line):
	return sum((ys_orig - ys_line)**2)

def coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for _ in ys_orig]
	squared_error_regression = squared_error(ys_orig, ys_line)
	squared_error_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (squared_error_regression / squared_error_mean)

def create_dataset(n, variance, step=2, correlation=None):
	val = 1
	ys = []
	for i in range(n):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step

	xs = [i for i in range(len(ys))]

	return np.array(xs), np.array(ys)

# xs = np.array([1,2,3,4,5])
# ys = np.array([5,4,6,5,6])

xs, ys = create_dataset(40, 40, correlation='pos')

m, b = best_fit_line(xs, ys)

regression_line = [m*x+b for x in xs]

r2 = coefficient_of_determination(ys, regression_line)
print(r2)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()