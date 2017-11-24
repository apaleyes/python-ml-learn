import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing

'''
List of columns in Titanic dataset

Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)

df.drop(['body', 'name', 'boat'], 1, inplace=True)
df = df.apply(lambda s: pd.to_numeric(s, 'ignore'))
df.fillna(0, inplace=True)

def handle_non_numerical_data(df):
	columns = df.columns.values
	for column in columns:
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)

			numeric_values = {}
			n = 0
			for unique in unique_elements:
				if unique not in numeric_values:
					numeric_values[unique] = n
					n+=1
			df[column] = list(map(lambda x: numeric_values[x], df[column]))
	return df

df = handle_non_numerical_data(df)
df.drop(['ticket','home.dest'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df["cluster_group"] = labels

n_clusters = len(cluster_centers)
survival_rates = {}
for i in range(n_clusters):
	temp_df = original_df[ (original_df["cluster_group"] == float(i)) ]
	survival_cluster = temp_df[ (temp_df["survived"]==1) ]
	survival_rate = len(survival_cluster) / len(temp_df)
	survival_rates[i] = survival_rate

print(survival_rates)