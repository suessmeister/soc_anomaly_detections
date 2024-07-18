# gathering all the imports

from tabulate import tabulate # for visualizing the pandas dataframe better. ipython can also be used?
import pandas as pd # for dataframe objects
import re # regular expression operations for strings.. used to tokenize the words without using vectors.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split # splitting the data accordingly

# Data Preprocessing - Making this data usable


raw_data = pd.read_excel("../incidents_jan_2024.xlsx") # 758 "rows" or incidents and 18 categorical columns

names = raw_data.iloc[:, 0] # get only the incident names for the dataframe

updated_data = pd.DataFrame()
updated_data['Raw Names'] = [str(names)] # add a column to this DataFrame storing the initial names. needs to be string.
updated_data['Lowercase Names'] = updated_data['Raw Names'].apply(lambda word: word.lower()) # make lowercase column.
print(updated_data.to_string())










