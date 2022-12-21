import pandas as pd
dt = pd.read_csv('E:\\machine learning\\airlines.csv')
# It is used fill missing value in integer form 
dt.interpolate()