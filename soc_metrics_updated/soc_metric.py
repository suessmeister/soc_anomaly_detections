import pandas as pd
import numpy as np
import incident


raw_incidents = pd.read_excel("incidents_jan_2024.xlsx")
processed_incidents = []

for index, raw_incident in raw_incidents.iterrows():
    df_array = raw_incident.to_numpy()

    # we'll need following indices: 0, 1, 3, 6, 16. these are the parameters needed to construct an incident object.
    indices = [0, 1, 3, 6, 16]

    processed_incident = incident.Incident(df_array[0], df_array[1], df_array[3], df_array[6], df_array[16])


    break


