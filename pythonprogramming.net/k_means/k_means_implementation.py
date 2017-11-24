import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11]])

class K_Means:
	def __init__(self, k=2, tol=0.001, max_iter=300):
		self.k = k
		self.tolerance = tol
		self.max_iter = max_iter

	def fit(self, data):
		np.random.shuffle(data)

		self.centroids = {}
		for i in range(self.k):
			self.centroids[i] = data[i]

		for i in range(self.max_iter):
			self.classifications = {}

			for j in range(self.k):
				self.classifications[j] = []

			for featureset in data:
				distances = [np.linalg.norm(featureset - self.centroids[c]) for c in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(featureset)

			prev_centroids = dict(self.centroids)

			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis=0)

			optimized = True
			for c in self.centroids:
				original_centroid = prev_centroids[c]
				current_centroid = self.centroids[c]
				if np.linalg.norm(current_centroid - original_centroid) / np.linalg.norm(original_centroid) * 100.0 > self.tolerance:
					optimized = False
					break

			if optimized:
				break

	def predict(self, data):
		distances = [np.linalg.norm(data - self.centroids[c]) for c in self.centroids]
		classification = distances.index(min(distances))
		return classification

clf = K_Means(k=3)
clf.fit(X)

print(clf.centroids)

for centroid in clf.centroids:
	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
				marker="o", color="k", s=150, linewidths=5)

colors = ["c", "b", "g", "r", "y", "m"]
for cl in clf.classifications:
	color = colors[cl]
	for featureset in clf.classifications[cl]:
		plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=100, linewidths=5)

unknowns = np.array([[2,3],
					 [10,10],
					 [5,1],
					 [1, 5],
					 [3,8],
					 [9, 6]])

for unknown in unknowns:
	cl = clf.predict(unknown)
	plt.scatter(unknown[0], unknown[1], marker="*", color=colors[cl], s=100, linewidths=5)

plt.show()