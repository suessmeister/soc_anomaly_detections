import pandas as pd
import numpy as np
import incident
import cfg


raw_incidents = pd.read_excel("incidents_jan_2024.xlsx")
processed_incidents = []

for index, raw_incident in raw_incidents.iterrows():
    df_array = raw_incident.to_numpy()

    # we'll need following indices: 0, 1, 3, 6, 16. these are the parameters needed to construct an incident object.
    indices = [0, 1, 3, 6, 16]

    # nice, these parameters work when incident string method is initialized!
    # define an initial threat index to 0 for each incident.
    processed_incident = incident.Incident(df_array[0], df_array[1], df_array[3],
                                           df_array[6], df_array[7],
                                           df_array[15], df_array[16],
                                           threat_index=0)


    break



