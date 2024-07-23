# gathering all the imports
import incident  # for defining the blueprint of an incident

# data preprocessing imports required
from tabulate import tabulate  # for visualizing the pandas dataframe better. ipython can also be used?
import pandas as pd  # for dataframe objects
import re  # regular expression operations for strings. used to tokenize the words without using vectors.
from nltk.corpus import stopwords  # for getting rid of the stopwords AKA words that are not needed for the model

from sklearn.feature_extraction.text import TfidfVectorizer  # assign tfidf scores in the model
from sklearn.neighbors import LocalOutlierFactor  # the machine learning model to use. this is unsupervised learning.

# Data Preprocessing - Making this data usable


raw_data = pd.read_excel("../incidents_jan_2024.xlsx")  # 758 "rows" or incidents and 18 categorical columns

# Defining the proper functions while also instantiating from the Incident Class.
def create_incidents():
    processed_incidents = []
    for index, raw_incident in raw_data.iterrows():
        df_array = raw_incident.to_numpy()

        # nice, these parameters work when incident string method is initialized!
        # define an initial threat index to 0 for each incident.
        processed_incident = incident.Incident(df_array[0], df_array[1], df_array[3],
                                               df_array[6], df_array[7],
                                               df_array[15], df_array[16],
                                               threat_index=0)

        processed_incidents.append(processed_incident)
    return processed_incidents


# Tokenizing the data. Used in lambda function call.
def tokenizer(str):
    tokens = re.split("\W+", str)
    return tokens


# Getting rid of stopwords while also removing unnecessary numbers (these should NOT be weighted).
# Used in lambda function call.
def clean_stopwords(token_list):  # getting rid of stopwords while also getting rid of tokens that start with numbers
    new_tokens = ""
    for token in token_list:
        if token not in stop_words and (not token[0].isdigit() if token else True):
            new_tokens = new_tokens + " " + token
    return new_tokens


# Data Preprocessing - lowercase, tokenizations, removing stopwords, vectorizing tokens, assigning TF_IDF scores

names = raw_data.iloc[:, 0]  # get only the incident names for the dataframe

updated_data = pd.DataFrame()
updated_data['Raw Names'] = names  # add a column to this DataFrame storing the initial names. needs to be string.
updated_data['Lowercase Names'] = updated_data['Raw Names'].apply(lambda word: word.lower())  # make lowercase column.

# add a column containing the tokens from each incident name. once again using lambda functions for simplicity.
updated_data['Tokenized Names'] = updated_data['Lowercase Names'].apply(lambda sentence: tokenizer(sentence))

stop_words = set(stopwords.words('english'))  # get the stopwords for the english dictionary

# NOTE!!! This requires you to open a Python Interpreter and use the following commands:
# import nltk
# nltk.download('stopwords')

# clean out words that do not contribute to the model
updated_data['No Stopwords'] = updated_data['Tokenized Names'].apply(lambda tokens: clean_stopwords(tokens))

vectorizer = TfidfVectorizer()
fitted_X = vectorizer.fit_transform(updated_data['No Stopwords'])  # learn the vocab and IDF scores


# Local Outlier Factor Model Creation and Fitting

model = LocalOutlierFactor()
model.fit(fitted_X)  # fit the model on the given data
outlier_scores = model.negative_outlier_factor_  # find local outliers. -1 is a constant value and indicates an inlier
updated_data['Outlier Scores'] = outlier_scores  # outliers further away from -1 are more likely to be outliers.
updated_data = updated_data.sort_values('Outlier Scores')  # descending order!


# Incident Matching or Data Preparation for Visualizations

# incidents now contains each incident in chronological order
incidents = create_incidents()




# print(tabulate(updated_data.head(), headers='keys', tablefmt='psql'))  # shoutout to tabulate for such a lightweight tool!
