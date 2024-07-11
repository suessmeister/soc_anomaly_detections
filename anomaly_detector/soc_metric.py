import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import incident
import cfg

# Preprocess the data, converting strings to integers and scaling for optimizations.
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
    print(df.to_string())
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


def main():
    create_incidents()
    preprocess_data()
    create_model()


main()
