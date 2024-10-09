import json
from State import State
from Action import Action
from Problem import Problem


def loadJSON(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    intersections = {}
    for i_data in data["intersections"]:
        inter = State(
            identifier=i_data["identifier"],
            latitude=i_data["latitude"],
            longitude=i_data["longitude"],
        )
        intersections[inter.identifier] = inter

    segments = []
    for seg_data in data["segments"]:
        origin = intersections[seg_data["origin"]]
        destination = intersections[seg_data["destination"]]
        segment = Action(
            origin=origin,
            destination=destination,
            distance=seg_data["distance"],
            speed=seg_data["speed"],
        )
        segments.append(segment)
        # Add neighbors to the origin state
        origin.neighbors.append((destination, segment))

    initial_state = intersections[data["initial"]]
    goal_state = intersections[data["final"]]

    return Problem(initial_state, goal_state, intersections, segments)
