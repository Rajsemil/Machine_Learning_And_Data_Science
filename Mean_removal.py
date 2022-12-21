import numpy as np
from sklearn import preprocessing 
input_data = np.array([[2.1, -1.9, 5.5],
	[-1.5, 2.4, 3.5],
 	[0.5, -7.9, 5.6],
 	[5.9, 2.3, -5.8]])
print("Mean =", input_data.mean(axis=0))
print("Stddeviation = ", input_data.std(axis=0))
data_scaled = preprocessing.scale(input_data)
print("Mean_removed =", data_scaled.mean(axis=0))
print("Stddeviation_removed =", data_scaled.std(axis=0))