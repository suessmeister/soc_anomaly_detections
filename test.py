# to test the different functionalities of ethan's code. DO NOT use this in the final version...

from IPython import display
import pandas as pd

# loading the raw incidents in the pandas form
raw_incidents = pd.read_excel("incidents_jan_2024.xlsx")

# print(f'Raw Incidents: {len(raw_incidents)}')
# print(f'Columns: {raw_incidents.columns}')

# get the unique incidents for every query, these are different policies/alerts that we get
unique_incidents = raw_incidents["Incident name"].unique()
print(unique_incidents)

DLP_incidents = []
for incident in unique_incidents:
    if 'DLP' in incident:
        DLP_incidents.append(incident)
del unique_incidents


# splitting up some of the lists so as to get a distinction in DLP incidents, maybe match email, document, cloud, or number of users
PII_DLP_incidents = []
PCI_DSS_incidents = []
leftover_incidents = []


for incident in DLP_incidents:
    if 'PII' in incident or 'SSN' in incident:
        PII_DLP_incidents.append(incident)
    elif 'PCI' in incident:
        PCI_DSS_incidents.append(incident)
    else:
        leftover_incidents.append(incident)



# print(f'PII: {PII_DLP_incidents}\n')
print(f'PCI: {PCI_DSS_incidents}\n')
print(f'leftovers: {leftover_incidents}')


# splitting the PII into further evaluation. 'ie.. is this email or document?'
email_PII_DLP = []
doc_PII_DLP = []
leftover_PII_DLP = []
for incident in PII_DLP_incidents:
    if 'email' in incident:
        email_PII_DLP.append(incident)
    elif 'document' in incident:
        doc_PII_DLP.append(incident)
    else:
        leftover_PII_DLP.append(incident)
print(f'Email PII DLP: {email_PII_DLP}\n')
print(f'Document PII DLP: {doc_PII_DLP}\n')
print(f'Leftovers: {leftover_PII_DLP}')

