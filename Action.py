class Action:
    def __init__(self, origin, destination, distance, speed):
        self.origin = origin  # State object
        self.destination = destination  # State object
        self.distance = distance  # Distance between origin and destination
        self.speed = speed / 3.6  # Speed limit on this segment

    def cost(self):
        # Calculate travel time as the cost
        return self.distance / self.speed

    def __repr__(self):
        return f"Action({self.origin.identifier} -> {self.destination.identifier}, distance={self.distance}, speed={self.speed})"
