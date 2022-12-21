import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd
Employee_Data = {
	'Nama':['Jatin Kumar', 'Sumt Saini', 'Aryan Khan', 'Raj Semil'],
	'ID':[121,232,432,432],
	'Age':[32,32,18,43],
	'salary':[23,45,212,54]
}
Employee_Data = pd.DataFrame(Employee_Data)
#Employee_Data.set_index(inplace=True)
Employee_Data
sns.clustermap(Employee_Data, figsize=(6,4), annot=True)
plt.show()