class Intersection:
    def __init__(self, identifier, latitude, longitude):
        self.id = identifier
        self.latitude = latitude
        self.longitude = longitude
        self.neighbours = (
            []
        )  # Adjacent intersection list (este es el caso que se comentaba en clase de guardar visitados creo)
