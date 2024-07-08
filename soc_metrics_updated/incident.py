# defining the blueprint for an incident object


class Incident:
    def __init__(self, name, id, severity, accounts_reached, impact, classification, determination, threat_index):
        self.name = name
        self.id = id
        self.severity = severity
        self.accounts_reached = accounts_reached
        self.impact = impact
        self.classification = classification
        self.determination = determination
        self.threat_index = threat_index


    def __str__(self):
        return (f"incident {self.name} with id {self.id} and severity {self.severity} and impact {self.impact}"
                f" with determination {self.determination}")

