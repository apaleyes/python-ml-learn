import numpy as np
import matplotlib.pyplot as plt
import warnings
from collections import Counter

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

def k_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is less than the number of classes')

	distances = []
	for group in data:
		for features in data[group]:
			distances.append((np.linalg.norm(np.array(features) - np.array(predict)), group))
	votes = [d[1] for d in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]

	return vote_result

result = k_neighbors(dataset, new_features)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=50, color=result)
plt.show()


