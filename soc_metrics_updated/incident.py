# defining the blueprint for creating an incident object, i.e. one row in the excel sheet.


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

    def calculate_threat_index(self):
        self.threat_index += 1
    def __str__(self):
        return (f"incident {self.name} with id {self.id} and severity {self.severity} and impact {self.impact}"
                f" with determination {self.determination}")

