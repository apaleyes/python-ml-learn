import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9, 11],
#               [8, 2],
#               [9, 2],
#               [9, 3]])

X, y = make_blobs(n_samples=15, centers=3, n_features=2)

class Mean_Shift:
	def __init__(self, radius=None, radius_steps=100):
		self.radius = radius
		self.radius_steps = radius_steps

	def fit(self, data):

		if self.radius is None:
			# This is not optimal as it depend on where points are related to (0,0)
			# TODO: optimize

			# shift all data to first quadrant to calculate radius
			shift_vector = np.amin(data, axis=0)
			shifted_data = data - shift_vector

			all_data_center = np.average(shifted_data, axis=0)
			center_norm = np.linalg.norm(all_data_center)
			self.radius = center_norm / self.radius_steps

		centroids = dict(enumerate(data))

		weights = list(range(self.radius_steps-1,-1,-1))
		while True:
			new_centroids = set()
			for i in centroids:
				# before intro of weights
				# in_bandwidth = []
				# centroid = centroids[i]
				# for featureset in data:
				# 	if np.linalg.norm(featureset - centroid) < self.radius:
				# 		in_bandwidth.append(featureset)

				weights_for_average = []
				centroid = centroids[i]
				for featureset in data:
					distance = np.linalg.norm(featureset - centroid)
					if distance == 0:
						distance = 0.000001
					weight_index = min(int(distance/self.radius), self.radius_steps-1)
					weight = weights[weight_index]**2
					weights_for_average.append(weight)

				new_centroid = np.average(data, axis=0, weights=weights_for_average)
				new_centroids.add(tuple(new_centroid))

			uniques = sorted(new_centroids)
			i = 0
			while i < len(uniques):
				# This does not reacognize really close centroids
				# Because radius can be even smaller
				# TODO: optimize
				uniques[i+1:] = [c for c in uniques[i+1:] if np.linalg.norm(np.array(uniques[i]) - np.array(c)) > self.radius]
				i+=1

			prev_centroids = dict(centroids)
			centroids = dict(enumerate(uniques))

			if len(centroids) != len(prev_centroids):
				continue

			optimized = True
			for i in centroids:
				if not np.array_equal(centroids[i], prev_centroids[i]):
					optimized = False
					break

			if optimized:
				break

		self.centroids = centroids

		self.classifications = {k: [] for k in range(len(self.centroids))}
		for featureset in data:
			distances = [np.linalg.norm(featureset - self.centroids[i]) for i in self.centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(featureset)

	def predict(self, featureset):
		distances = [np.linalg.norm(featureset - self.centroids[i]) for i in self.centroids]
		classification = distances.index(min(distances))
		return classification

clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids
for i in centroids:
	plt.scatter(centroids[i][0], centroids[i][1], color="k", marker="x", s=150, linewidths=5)

colors = 10*["y", "b", "m", "g", "r", "c"]
for c in clf.classifications:
	color = colors[c]
	for featureset in clf.classifications[c]:
		plt.scatter(featureset[0], featureset[1], color=color, marker="o", s=150)

plt.show()