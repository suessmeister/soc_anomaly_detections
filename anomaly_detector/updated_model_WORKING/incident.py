# defining the blueprint for creating an incident object, i.e. one row in the excel sheet.
# has different methods depending on current and/or future needs
# also contains functions to modify
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

    def return_raw_data(self):
        return [
            self.severity,
            self.impact,
            self.name
        ]


    def __str__(self):
        return (f"incident {self.name} with id {self.id} and severity {self.severity} and impact {self.impact}"
                f" with determination {self.determination}")

