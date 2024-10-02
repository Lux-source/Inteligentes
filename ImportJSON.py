import json
from Implementation.Inteligentes.Intersection import Intersection


def loadJSON(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    intersections = {}
    for i_data in data["intersections"]:
        inter = Intersection(
            identifier=i_data["identifier"],
            latitude=i_data["latitude"],
            longitude=i_data["longitude"],
        )
        intersections[inter.id] = inter
