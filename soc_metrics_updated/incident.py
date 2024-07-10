# defining the blueprint for creating an incident object, i.e. one row in the excel sheet.
from sklearn.preprocessing import LabelEncoder
class Incident:
    def __init__(self, name, id, severity, accounts_reached, impact, classification, determination, threat_index):
        self.name = name
        self.id = id
        self.severity = severity
        self.accounts_reached = accounts_reached
        self.impact = impact
        self.classification = classification
        self.determination = determination

        # real/unknown incidents marked as a '1' else do not mark as an active incident.
        self.threat_index = 1 if self.classification in {"True alert", "Benign Positive", "Not set"} else 0


    def __encode_data(self):
        # severity can be split up into low, medium, or high. so we need a categorical encoder! order matters.
        encoder = LabelEncoder()

        # encode labels for "severity"
        self.severity_encoded = encoder.fit_transform(self.severity)

        # encode labels for "determination"
        self.determination_encoded = encoder.fit_transform(self.determination)
    def __prepare_data(self):

        data = [
            self.severity_encoded,
            len(int(self.impact))

        ]


    def __str__(self):
        return (f"incident {self.name} with id {self.id} and severity {self.severity} and impact {self.impact}"
                f" with determination {self.determination}")

