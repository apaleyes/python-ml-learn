import pandas as pd
import numpy as np

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
print(df.head())
