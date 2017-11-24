import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import sys
style.use('ggplot')

class SupportVectorMachine:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1: 'r', -1: 'b', 0: 'g'}
		if self.visualization:
			self.figure = plt.figure()
			self.ax = self.figure.add_subplot(1, 1, 1)

	def fit(self, data):
		self.data = data
		opt_dict = {}

		transforms = [[1, 1], [1, -1], [-1, -1], [-1, 1]]

		self.max_feature_value = -sys.maxsize
		self.min_feature_value = sys.maxsize
		for yi in self.data:
			for featureset in self.data[yi]:
				self.max_feature_value = max(self.max_feature_value, max(featureset))
				self.min_feature_value = min(self.min_feature_value, min(featureset))

		step_sizes = [self.max_feature_value * 0.1,
					  self.max_feature_value * 0.01,
					  # high cost after this
					  self.max_feature_value * 0.001
					  ]

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
								if yi * (np.dot(w_t, xi) + b) < 1:
									found_option = False
									break
							if not found_option:
								break
						if found_option:
							opt_dict[np.linalg.norm(w_t)] = [w_t, b]

				if w[0] < 0:
					optimized = True
					print("Optimized a step: {}".format(step))
				else:
					w = w - step

			opt_norm = min([n for n in opt_dict])
			opt_choice = opt_dict[opt_norm]
			self.w = opt_choice[0]
			self.b = opt_choice[1]
			latest_optimum = opt_choice[0][0] + step * 2


	def predict(self, features):
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		if self.visualization:
			self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
		return classification

	def visualize(self):
		[[self.ax.scatter(x[0], x[1], s=100, c=self.colors[i]) for x in self.data[i]] for i in self.data]

		# v = x.w + b
		def hyperplane(x, w, b, v):
			return (-w[0]*x-b+v)/w[1]

		def plot_hyperplane(x1, x2, w, b, v, style):
			pt1 = hyperplane(x1, w, b, v)
			pt2 = hyperplane(x2, w, b, v)
			self.ax.plot([x1, x2], [pt1, pt2], style)

		hyp_x_min = np.sign(self.min_feature_value)*abs(self.min_feature_value)*1.1
		hyp_x_max = self.max_feature_value*1.1

		# posivite support vector hyperplane
		plot_hyperplane(hyp_x_min, hyp_x_max, self.w, self.b, 1, 'k')
		# negative support vector hyperplane
		plot_hyperplane(hyp_x_min, hyp_x_max, self.w, self.b, -1, 'k')
		# decision boundary
		plot_hyperplane(hyp_x_min, hyp_x_max, self.w, self.b, 0, 'y--')
		
		axes = plt.gca()
		axes.set_xlim([hyp_x_min,hyp_x_max])
		axes.set_ylim([hyp_x_min,hyp_x_max])
		plt.show()

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}

svm = SupportVectorMachine()
svm.fit(data_dict)

predict_us = [[0,10],
              [1,3],
              [3,4],
              [3,5],
              [5,5],
              [5,6],
              [6,-5],
              [5,8]]
for p in predict_us:
	svm.predict(p)

svm.visualize()