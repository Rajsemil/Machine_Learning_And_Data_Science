import pandas as pd
dt = pd.read_csv('E:\\machine learning\\airlines.csv')
# describe is used to show in leveling form of data
"""
count
mean
std
min
25%
50%
75%
max
"""
dt.describe()