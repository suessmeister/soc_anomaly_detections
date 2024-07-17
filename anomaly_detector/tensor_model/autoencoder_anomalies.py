# gathering all the imports
import pandas as pd # for dataframe objects

from sklearn.model_selection import train_test_split # splitting the data accordingly


# Data Preprocessing - Making this data usable
raw_data = pd.read_excel("../incidents_jan_2024.xlsx") # 758 "rows" or incidents and 18 categorical columns

names = raw_data.iloc[:, 0] # get only the incident names for the dataframe
print(names)






