import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2],
	          [2, 1],
	          [5, 8],
	          [8, 8],
	          [1, 0.5],
	          [9, 10]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()

clf = KMeans(n_clusters=6)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = ['b.', 'r.', 'g.', 'y.', 'c.', 'm.']
for i in range(len(X)):
	plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, zorder=10, color='k')
plt.show()