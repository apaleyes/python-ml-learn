import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
styles.use('ggplot')

class SupportVectorMachine:
	def __init__(self, visualization=True):
		self.visualization = visualization
		self.colors = {1: 'r', -1: 'b'}
		if self.visualization:
			self.figure = plt.figure()
			self.ax = self.figure.add_subplot(1, 1, 1)

	def fit(self, data):
		pass

	def predict(self, features):
		classification = np.sign(np.dot(np.array(features), self.w) + self.b)
		return classification