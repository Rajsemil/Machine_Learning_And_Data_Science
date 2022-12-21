import pandas as pd
# Merge is used to bind multi files 
dt = pd.DataFrame({'ID':[122,34,53,75,23,45,21],
                  'NAME':['Sameer', 'Rajeev', 'Ram','Raj', 'Shivam', 'Vikas', 'Pritee']})
print(dt)
dt1 = pd.DataFrame({'ID':[12,4,53,75,23,45,21],
                  'PROGRAMMER':['C', 'C++', 'Rubby','Java', 'Python', 'PHP', 'JavaScript']})
print(dt1)
# on method is used to same data in different files
print(pd.merge(dt, dt1, on='ID'))
print(pd.merge(dt1, dt, on='ID'))
print(pd.merge(dt, dt1, on='ID', how = 'right'))
print(pd.merge(dt, dt1, on='ID', how = 'left'))
print(pd.merge(dt, dt1, on='ID', how = 'inner'))
print(pd.merge(dt, dt1, on='ID', how = 'outer'))
# indicator identified data such that same name colum in different files but value will be different 
print(pd.merge(dt, dt1, on='ID', how = 'left', indicator = True))