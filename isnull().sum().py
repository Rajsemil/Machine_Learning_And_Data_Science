import pandas as pd
dt = pd.read_csv('E:\\machine learning\\airlines.csv')
# That is used for check missing the values in the data
dt.isnull().sum()