# gathering all the imports

# data preprocessing imports required
from tabulate import tabulate # for visualizing the pandas dataframe better. ipython can also be used?
import pandas as pd # for dataframe objects
import re # regular expression operations for strings.. used to tokenize the words without using vectors.
from nltk.corpus import stopwords # for getting rid of the stopwords AKA words that are not needed for the model



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split # splitting the data accordingly

# Data Preprocessing - Making this data usable


raw_data = pd.read_excel("../incidents_jan_2024.xlsx") # 758 "rows" or incidents and 18 categorical columns

names = raw_data.iloc[:, 0] # get only the incident names for the dataframe

updated_data = pd.DataFrame()
updated_data['Raw Names'] = names # add a column to this DataFrame storing the initial names. needs to be string.
updated_data['Lowercase Names'] = updated_data['Raw Names'].apply(lambda word: word.lower()) # make lowercase column.

def tokenizer(str):
    tokens = re.split("\W+", str)
    return tokens


# add a column containing the tokens from each incident name. once again using lambda functions for simplicity.
updated_data['Tokenized Names'] = updated_data['Lowercase Names'].apply(lambda sentence: tokenizer(sentence))

stop_words = set(stopwords.words('english')) # get the stopwords for the english dictionary

# NOTE!!! This requires you to open a Python Interpreter and use the following commands:
# import nltk
# nltk.download('stopwords')

def clean_stopwords(token_list):
    new_tokens = []
    for token in token_list:
        if token not in stop_words: new_tokens.append(token)
    return new_tokens


# clean out words that do not contribute to the model
updated_data['No Stopwords'] = updated_data['Tokenized Names'].apply(lambda tokens: clean_stopwords(tokens))



print(tabulate(updated_data, headers='keys', tablefmt='psql')) # shoutout to tabulate for such a lightweight tool!











