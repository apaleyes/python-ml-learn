import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import sys
styles.use('ggplot')

class SupportVectorMachine:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1: 'r', -1: 'b'}
		if self.visualization:
			self.figure = plt.figure()
			self.ax = self.figure.add_subplot(1, 1, 1)

	def fit(self, data):
		self.data = data
		opt_dict = {}

		transforms = [[1, 1], [1, -1], [-1, -1], [-1, 1]]

		self.max_feature_value = sys.maxsize
		self.min_feature_value = -sys.maxsize
		for yi in self.data:
			for featureset in self.data[yi]:
				self.max_feature_value = max(self.max_feature_value, max(featureset))
				self.min_feature_value = min(self.min_feature_value, min(featureset))

		step_sizes = [self.max_feature_value * 0.1,
					  self.max_feature_value * 0.01,
					  # high cost after this
					  self.max_feature_value * 0.001]

		# extremely expensive
		b_range_multiple = 5
		b_multiple = 5
		latest_optimum = self.max_feature_value * 10

		for step in step_sizes:
			w = np.array([latest_optimum, latest_optimum])
			optimized = False
			while not optimized:
				b_range = np.arange(-1 * (self.max_feature_value * b_range_multiple),
					                self.max_feature_value * b_range_multiple,
					                step * b_multiple)
				for b in b_range:
					for transform in transforms:
						w_t = w * transform
						found_option = True
						for yi in data:
							for xi in data[yi]:
								if yi * (np.dot(w_t, xi) + b) < 1
									found_option = False
									break
							if not found_option:
								break
						if found_option:
							opt_dict[np.lingalg.norm(w_t)] = [w_t, b]

				if w[0] < 0:
					optimized = True
					print("Optimized a step: " + step)
				else:
					w = w - step

		opt_choice = max([n for n in opt_dict])
		self.w = opt_choice[0]
		self.b = opt_choice[1]
		latest_optimum = opt_choice[0][0] + step * 2


	def predict(self, features):
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		return classification