# gathering all the imports

# data preprocessing imports required
from tabulate import tabulate # for visualizing the pandas dataframe better. ipython can also be used?
import pandas as pd # for dataframe objects
import re # regular expression operations for strings.. used to tokenize the words without using vectors.
from nltk.corpus import stopwords # for getting rid of the stopwords AKA words that are not needed for the model



from sklearn.feature_extraction.text import TfidfVectorizer # assign tfidf scores in the model
from sklearn.model_selection import train_test_split # splitting the data accordingly
from sklearn.neighbors import LocalOutlierFactor

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

def clean_stopwords(token_list): # getting rid of stopwords while also getting rid of tokens that start with numbers
    new_tokens = ""
    for token in token_list:
        if token not in stop_words and (not token[0].isdigit() if token else True):
            new_tokens = new_tokens + " " + token
    return new_tokens


# clean out words that do not contribute to the model
updated_data['No Stopwords'] = updated_data['Tokenized Names'].apply(lambda tokens: clean_stopwords(tokens))


vectorizer = TfidfVectorizer()
fitted_X = vectorizer.fit_transform(updated_data['No Stopwords']) # Learn and Vocab AND Return IDF scores
print(vectorizer.get_feature_names_out())
print(vectorizer.idf_)

# X_train, X_test = train_test_split(fitted_X, test_size=0.15)

model = LocalOutlierFactor()

model.fit(fitted_X)

outlier_scores = model.negative_outlier_factor_

updated_data['Outlier Scores'] = outlier_scores


print(tabulate(updated_data, headers='keys', tablefmt='psql')) # shoutout to tabulate for such a lightweight tool!











