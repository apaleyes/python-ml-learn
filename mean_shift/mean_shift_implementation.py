import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [9, 2],
              [9, 3]])

class Mean_Shift:
	def __init__(self, radius=4):
		self.radius = radius

	def fit(self, data):
		centroids = dict(enumerate(data))

		while True:
			new_centroids = set()
			for i in centroids:
				in_bandwidth = []
				centroid = centroids[i]
				for featureset in data:
					if np.linalg.norm(featureset - centroid) < self.radius:
						in_bandwidth.append(featureset)

				new_centroid = np.average(in_bandwidth, axis=0)
				new_centroids.add(tuple(new_centroid))

			uniques = sorted(new_centroids)
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

	def predict(self, data):
		pass

clf = Mean_Shift()
clf.fit(X)

plt.scatter(X[:, 0], X[:, 1], color="b", s=150)
centroids = clf.centroids
for i in centroids:
	plt.scatter(centroids[i][0], centroids[i][1], color="k", marker="x", s=150, linewidths=5)

plt.show()