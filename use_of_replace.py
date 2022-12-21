import pandas as pd
dt = pd.read_csv('E:\\machine learning\\airlines.csv')
# use of value replace
dt.replace('Name', 'Value')
# for integer value replace
dt.replace(10, 20)
# for colum replace from one value
dt.replace([1,2,3,4,5,6,7,8],0)
# for defferent value replace
dt.replace([1,2,3,4,5,6,7,8],[10,43,56,44,65,21,64,88]