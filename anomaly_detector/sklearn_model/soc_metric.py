import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import incident

# Preprocess the data, converting strings to integers and scaling for optimizations.
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Preprocessing, continued.
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# For use finding the anomalies.
from sklearn.neighbors import LocalOutlierFactor

# Clean up the logic for the steps to reproduce the model.
from sklearn.pipeline import Pipeline

# hyper-optimization on model parameters
from sklearn.model_selection import GridSearchCV

global df
raw_data = []
raw_incidents = pd.read_excel("incidents_jan_2024.xlsx")
processed_incidents = []


def create_incidents():
    for index, raw_incident in raw_incidents.iterrows():
        df_array = raw_incident.to_numpy()

        # nice, these parameters work when incident string method is initialized!
        # define an initial threat index to 0 for each incident.
        processed_incident = incident.Incident(df_array[0], df_array[1], df_array[3],
                                               df_array[6], df_array[7],
                                               df_array[15], df_array[16],
                                               threat_index=0)
        if processed_incident.threat_index == 1:
            raw_data.append(processed_incident.return_raw_data())


def preprocess_data():
    global df
    df = pd.DataFrame(raw_data)
    encoder = LabelEncoder()
    df.iloc[:, 0] = encoder.fit_transform(df.iloc[:, 0])

    # now for funsies, let's graph this data so we might find some anomalies using our eyes alone.
    # print(df.to_string())
    plt.figure()
    df.plot()


def create_model():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", LocalOutlierFactor())
    ])

    # find potential parameters in the pipeline.
    print(pipe.get_params())

    parameter_grid = {
        'model__n_neighbors': np.arange(start=1, stop=11, step=1).tolist(),
        'model__contamination': np.arange(start=0.1, stop=0.5, step=0.05).tolist()
    }

    grid = GridSearchCV(
        estimator='pipe'
    )
    # now fit to the data.
    pipe.fit(df)

    # get the outlier scores for the model
    outlier_scores = pipe.named_steps['model'].negative_outlier_factor_

    # define what an outlier is
    outliers = df[outlier_scores < -3]

    print(outliers)

    # print(scores)
    # outliers = np.argwhere(scores > np.percentile(scores, 95))
    # print(outliers)
    # colors = ['green', 'red']
    # plt.show()

    # for i in range(len(df)):
    #     if i not in outliers:
    #         plt.scatter(df.iloc[i, 0], df.iloc[i, 1], color=colors[0])  # Not anomly
    #     else:
    #         plt.scatter(df.iloc[i, 0], df.iloc[i, 1], color=colors[1])  # anomly
    # plt.title('Local Outlier Factor', fontsize=16)
    # plt.show()
    # return 0

def better_model():

    # in the previous model, we do NOT vectorize the name of the incident.
    # however, this could yield some very powerful results and give us specific anomalies.

    # consider a vector space in the R3 space.
    # a vector representation of an incident could have the tokenized name on the x,
    # accounts impacted on the y,
    # and finally the severity on the z.
    # the total magnitude of this vector will yield far better results.

    names = df.iloc[:, 2]

    # should be email reported by one user if using incidents jan 2024... testing if df is properly fitted
    print(names[0])

    # initializing the count vectorizer!
    cv = CountVectorizer()

    # generating the word counts for the words.
    word_count_vector = cv.fit_transform(names)

    # printing (data entries, amount of unique words)
    print(word_count_vector.shape)

    # optional: get the actual dictionary
    dict = cv.vocabulary_

    # and a little java-ish syntax to print nicely. do not pay much attention to the second values (idetities)
    # print("\n".join(f"{i, j}" for i, j in dict.items()))

    # computing the IDF values.
    # we need to call the IDF Transformer and find the "weights" for the matrix
    # lower the IDF --> less unique or more common. inverse relationship!

    transformer = TfidfTransformer() # lots of linear algebra in sklearn's implementation :)

    # fit transformer on word count vector
    transformer.fit(word_count_vector)

    idf_values = pd.DataFrame(transformer.idf_, index=cv.get_feature_names_out(), columns=["weights"])

    idf_values.sort_values(ascending=False, by=["weights"])

    # print(idf_values.to_string())

    # once the idf values are found, we can now compute the tfidf scores
    count_vector = cv.transform(names)

    # tf-idf scores
    tf_vector = transformer.transform(count_vector)

    # the first line gets the word counts in the given file in a sparse matrix. the word count vector could
    # have also been used, but for debugging purposes, we can use this on any xl sheet.
    # then, we compute the tfidf scores by computing tf*idf multiplication where the weights are applied.
    # all that is left is to print these values, so let's create this as a new object

    feature_names = cv.get_feature_names_out()

    final_vector = tf_vector[0]

    updated_df = pd.DataFrame(final_vector.T.todense(), index=feature_names, columns=["tfidf"])
    updated_df.sort_values(ascending=False, by=["tfidf"])

    model = LocalOutlierFactor()
    predictions = model.fit_predict(updated_df)
    print(updated_df.to_string())
    print(predictions)
    print(model.negative_outlier_factor_)

def main():
    create_incidents()
    preprocess_data()
    # create_model()
    better_model()


main()
