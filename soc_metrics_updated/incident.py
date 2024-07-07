# defining the blueprint for an incident
# an incident needs a category
class Incident:
    def __init__(self, name, id, severity, impact, determination):
        self.name = name
        self.id = id
        self.severity = severity
        self.impact = impact
        self.determination = determination

    def __str__(self):
        return f"incident {self.name}..."