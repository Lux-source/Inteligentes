class State:
    def __init__(self, identifier, latitude, longitude):
        self.identifier = identifier
        self.latitude = latitude
        self.longitude = longitude
        self.neighbors = []  # List of (neighbor_state, action/segment) tuples

    def __eq__(self, other):
        return isinstance(other, State) and self.identifier == other.identifier

    def __hash__(self):
        # Hash method to allow using State instances in sets or as dictionary keys
        return hash(self.identifier)

    def __repr__(self):
        return f"State({self.identifier}, {self.latitude}, {self.longitude})"
